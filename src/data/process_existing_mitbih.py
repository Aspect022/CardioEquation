import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.mitbih_long_loader import MITBIHLongLoader

loader = MITBIHLongLoader()
# Skip download, just process existing
loader.process()
