from typing import Optional, Tuple
from torch import Tensor

import collections

Rays = collections.namedtuple("Rays", ("origins", "directions"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

import torch
import torch_scatter


def compute_delta(a : Tensor, b : float, c : float):
    return b**2 - 4*a*c

def r1(a : Tensor, b : float, delta : Tensor) -> Tensor:
    return (-b - torch.sqrt(delta)) / (2 * a)

def r2(a : Tensor, b : float, delta : Tensor) -> Tensor:
    return (-b + torch.sqrt(delta)) / (2 * a)

def ray_intersect(
    source_to_obj: float,
    source_to_det: float,
    reconstruction_radius: float,
    ray_length: Tensor) -> Tuple[Tensor, Tensor]:
    
    b = -2 * source_to_obj * source_to_det
    c = source_to_obj**2 - reconstruction_radius**2
    a = ray_length**2 # [num_rays, 1]
    
    # ray_length = torch.norm(ray_directions, dim=-1, keepdim=True) # [num_rays, 1]

    # ray square norm = square of ray length
    delta = compute_delta(a, b, c)

    t_start = ray_length * r1(a, b ,delta) # [num_rays, 1]
    t_end = ray_length * r2(a, b ,delta)  # [num_rays, 1]
    
    return t_start, t_end

def sample_and_integrate(
    # scene
    neural_field,
    estimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    t_min: Optional[Tensor] = None,  # [n_rays]
    t_max: Optional[Tensor] = None,  # [n_rays]
    render_step_size: float = 1e-3,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    if t_min is not None:
        t_min = torch.where(torch.isnan(t_min), torch.tensor(near_plane, dtype=t_min.dtype), t_min)
    if t_max is not None:
        t_max = torch.where(torch.isnan(t_max), torch.tensor(far_plane, dtype=t_max.dtype), t_max)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if neural_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.directions
        
        t_min_chunk = t_min[i : i + chunk] if t_min is not None else None
        t_max_chunk = t_max[i : i + chunk] if t_max is not None else None

        def sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )

                sigmas = neural_field(positions)
            return sigmas.squeeze()
        
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            t_min=t_min_chunk,
            t_max=t_max_chunk,
            render_step_size=render_step_size,
            stratified=neural_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        attenuation = sigma_fn(t_starts, t_ends, ray_indices)
        output_integration = torch_scatter.scatter(attenuation, 
                                    ray_indices, 
                                    dim=0, 
                                    reduce='sum', 
                                    dim_size=rays_o.size(0)) * render_step_size
        
        chunk_results = [output_integration, len(t_starts)]
        results.append(chunk_results)

    output, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        output.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples)
    )