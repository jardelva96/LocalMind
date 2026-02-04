import os
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def test_predict_csv_cli():
    """Test the predict_csv CLI with a temporary CSV file."""
    # Create a temporary CSV with sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        # Create sample data with 32 features (matching test_api.py)
        df = pd.DataFrame(np.random.rand(10, 32), columns=[f'feature_{i}' for i in range(32)])
        df.to_csv(csv_path, index=False)
    
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            out_path = f.name
        
        try:
            # Get the API URL from environment or use default
            base_url = os.getenv("IAZERO_URL", "http://127.0.0.1:8000")
            api_key = os.getenv("IAZERO_API_KEY", None)
            
            # Build command
            cmd = [
                "python", "-m", "iac_core.predict_csv",
                "--csv", csv_path,
                "--out", out_path,
                "--url", f"{base_url}/predict",
            ]
            
            if api_key:
                cmd.extend(["--api-key", api_key])
            
            # Run the CLI command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if command succeeded
            assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
            
            # Verify output file was created
            assert Path(out_path).exists(), "Output CSV was not created"
            
            # Load and verify output
            out_df = pd.read_csv(out_path)
            assert len(out_df) == 10, "Output CSV should have 10 rows"
            assert "pred" in out_df.columns, "Output CSV should have 'pred' column"
            
        finally:
            # Clean up output file
            if Path(out_path).exists():
                Path(out_path).unlink()
    
    finally:
        # Clean up input file
        if Path(csv_path).exists():
            Path(csv_path).unlink()
