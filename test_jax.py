# In ~/jax_test_venv/minimal_test.py

try:
    print("Attempting to import flax...")
    from flax.training import checkpoints
    print("Successfully imported flax.training.checkpoints")
    
    print("\nAttempting to import jax.experimental.maps...")
    # This is the line that fails in your (dmac) environment
    from jax.experimental import maps
    print("Successfully imported jax.experimental.maps")
    
    print("\n✅ Success! This JAX/Flax installation is internally consistent.")

except ImportError as e:
    print(f"\n❌ Failure! An import error occurred: {e}")
    print("This indicates a version mismatch between JAX and Flax.")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")