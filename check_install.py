import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
try:
    import groundingdino, sam2
    print("GroundingDINO import: ok")
    print("SAM 2 import: ok")
except Exception as e:
    print("Import note:", e)
print("Done")
