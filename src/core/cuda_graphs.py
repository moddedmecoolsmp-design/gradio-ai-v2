"""CUDA Graph runner for accelerating repeated diffusion inference.

Captures the UNet/transformer execution into a CUDA graph for replay,
eliminating CPU kernel launch overhead. Best for fixed input shapes.
"""

import torch


class CUDAGraphRunner:
    """CUDA Graph runner for accelerating repeated diffusion inference."""

    def __init__(self):
        self.graphs = {}  # (batch, height, width, steps) -> (graph, static_inputs, static_output)
        self.warmup_done = False

    def _make_static_inputs(self, pipe, batch_size, height, width, device, dtype):
        """Create static input tensors for graph capture."""
        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        if hasattr(pipe, "transformer"):
            in_channels = getattr(pipe.transformer, "in_channels", 16)
            latents = torch.randn(
                (batch_size, in_channels, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )
        else:
            latents = torch.randn(
                (batch_size, 4, latent_height, latent_width),
                device=device,
                dtype=dtype,
            )

        return latents

    def capture_graph(self, pipe, batch_size, height, width, steps, device="cuda"):
        """Capture a CUDA graph for given dimensions."""
        if device != "cuda" or not torch.cuda.is_available():
            return False

        key = (batch_size, height, width, steps)
        if key in self.graphs:
            return True

        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            return False

        print(f"  [CUDA Graph] Capturing graph for {height}x{width} @ {steps} steps...")

        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(s):
                for _ in range(3):
                    static_inputs = self._make_static_inputs(
                        pipe, batch_size, height, width, device, torch.float16
                    )
                    if hasattr(pipe, "transformer"):
                        pipe.transformer(
                            static_inputs,
                            timestep=1.0,
                            encoder_hidden_states=torch.randn(
                                1, 77, 4096, device=device, dtype=torch.float16
                            ),
                        )
                    elif hasattr(pipe, "unet"):
                        pipe.unet(
                            static_inputs,
                            1.0,
                            encoder_hidden_states=torch.randn(
                                1, 77, 2048, device=device, dtype=torch.float16
                            ),
                        )

            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            static_inputs = self._make_static_inputs(
                pipe, batch_size, height, width, device, torch.float16
            )

            with torch.cuda.graph(g):
                if hasattr(pipe, "transformer"):
                    static_output = pipe.transformer(
                        static_inputs,
                        timestep=1.0,
                        encoder_hidden_states=torch.randn(
                            1, 77, 4096, device=device, dtype=torch.float16
                        ),
                    )
                elif hasattr(pipe, "unet"):
                    static_output = pipe.unet(
                        static_inputs,
                        1.0,
                        encoder_hidden_states=torch.randn(
                            1, 77, 2048, device=device, dtype=torch.float16
                        ),
                    )

            self.graphs[key] = (g, static_inputs, static_output)
            print(f"  [CUDA Graph] Captured successfully for {height}x{width}")
            return True

        except Exception as e:
            print(f"  [CUDA Graph] Capture failed: {e}")
            return False

    def replay(self, batch_size, height, width, steps):
        """Replay a captured graph. Returns (success, output)."""
        key = (batch_size, height, width, steps)
        if key not in self.graphs:
            return False, None

        g, static_inputs, static_output = self.graphs[key]
        g.replay()
        return True, static_output

    def is_captured(self, batch_size, height, width, steps):
        """Check if graph is captured for given dimensions."""
        return (batch_size, height, width, steps) in self.graphs


# Global CUDA graph runner (lazy init)
_cuda_graph_runner = None


def get_cuda_graph_runner():
    """Get or create the global CUDA graph runner."""
    global _cuda_graph_runner
    if _cuda_graph_runner is None:
        _cuda_graph_runner = CUDAGraphRunner()
    return _cuda_graph_runner
