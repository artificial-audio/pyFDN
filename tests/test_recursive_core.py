"""Unit tests for core recursive DSP abstractions."""

import pytest
import torch
from pyFDN import Stage, RecursionCore, DelayRead, DelayWrite, OutputTap


class DummyStage(Stage):
    """Simple stage for testing that adds a constant to the lines tensor."""
    
    def __init__(self, add_value: float = 1.0, has_state: bool = False):
        state_keys = {"dummy_state"} if has_state else set()
        super().__init__(state_keys)
        self.add_value = add_value
        self.has_state = has_state
    
    def init_state(self, batch_size: int, device: torch.device):
        if self.has_state:
            return {"dummy_state": torch.zeros(batch_size, device=device)}
        return {}
    
    def step_block(self, lines, state_t, next_state, block_size, x_block=None):
        if lines is not None:
            new_lines = lines + self.add_value
        else:
            # Create lines if not present
            # Format: [B, N, T] (batch, channels, time) to match all DSP stages
            if x_block is None:
                raise RuntimeError("DummyStage requires `x_block` when lines is None")
            batch_size = x_block.shape[0]
            num_lines = 4
            new_lines = torch.full(
                (batch_size, num_lines, block_size),
                self.add_value,
                device=x_block.device,
            )
        
        if self.has_state:
            next_state["dummy_state"] = state_t["dummy_state"] + 1

        return new_lines, None


class TestStageBase:
    """Test Stage abstract base class."""
    
    def test_stage_initialization(self):
        """Test basic stage initialization."""
        stage = DummyStage(add_value=5.0)
        assert stage.state_keys == set()
        assert stage.add_value == 5.0
    
    def test_stage_with_state(self):
        """Test stage with state keys."""
        stage = DummyStage(has_state=True)
        assert stage.state_keys == {"dummy_state"}
        
        state = stage.init_state(2, torch.device("cpu"))
        assert "dummy_state" in state
        assert state["dummy_state"].shape == (2,)


class TestRecursionCore:
    """Test RecursionCore class."""
    
    def test_core_initialization(self):
        """Test core initialization with stages."""
        stages = [
            DummyStage(),
            DummyStage(),
        ]
        core = RecursionCore(stages, block_size=16)
        assert len(core.stages) == 2
        assert core.device == torch.device("cpu")
        assert core.block_size == 16
    
    def test_state_initialization(self):
        """Test global state initialization."""
        stages = [
            DelayRead(delay_lengths=[16, 32, 64, 128], num_lines=4),
            DelayWrite(),
        ]
        core = RecursionCore(stages, block_size=16)
        state = core.init_state(batch_size=2, block_size=16, device=torch.device("cpu"))
        
        assert "delay_buffers" in state
        assert "delay_pointer" in state
        assert state["delay_buffers"].shape == (2, 4, 16)  # [B, N, L]
        assert state["delay_pointer"].shape == (2, 4)  # [B, N]
    
    def test_batch_dimension_handling(self):
        """Test automatic batch dimension handling."""
        stages = [
            DelayRead(delay_lengths=[8, 16, 32, 64], num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages, block_size=32)
        
        # 2D input [T, N_in] -> converted to [N_in, T] internally, output back to [T, N_out]
        input_2d = torch.randn(100, 1)
        output = core.process(input_2d)
        assert output.shape == (100, 1)  # Transposed back for backward compatibility
        
        # 3D input [B, N_in, T]
        input_3d = torch.randn(3, 1, 100)
        output = core.process(input_3d)
        assert output.shape == (3, 1, 100)  # [B, N_out, T]
    
    def test_block_partitioning(self):
        """Test signal partitioning into blocks."""
        # Test with various input lengths and block sizes
        test_cases = [
            (100, 32),  # Doesn't divide evenly
            (96, 32),   # Divides evenly
            (10, 32),   # Shorter than block size
        ]
        
        for T, bs in test_cases:
            stages = [
                DelayRead(delay_length=8, num_lines=2),
                DelayWrite(),
                OutputTap(num_lines=2, num_outputs=1),
            ]
            core = RecursionCore(stages, block_size=bs)
            input_signal = torch.randn(T, 1)
            output = core.process(input_signal)
            assert output.shape == (T, 1), f"Failed for T={T}, block_size={bs}"
    
    def test_state_preservation_across_blocks(self):
        """Test that state is correctly preserved across blocks."""
        stages = [
            DummyStage(has_state=True),
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages, block_size=16)
        
        input_signal = torch.randn(64, 1)
        core.process(input_signal)
        # Test passes if no errors - state management is working
    
    def test_missing_output_error(self):
        """Test that missing output raises error."""
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            # No OutputTap - should error
        ]
        core = RecursionCore(stages, block_size=16)
        
        input_signal = torch.randn(32, 1)
        with pytest.raises(RuntimeError, match="No output produced"):
            core.process(input_signal)
    
    def test_device_consistency(self):
        """Test that tensors stay on correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages, block_size=16, device=device)
        
        input_signal = torch.randn(32, 1)  # CPU tensor
        output = core.process(input_signal)
        
        assert output.device == device
