Edits were made on matgl/apps/pes.py

This script allows the use of M3GNet and other models in MatPES group (https://matpes.ai/) for Potential eneregy calculation. 

What our script does: 
1. Turn off the use of triplet by use_edge=False
2. Apply chunking process by setting apply_chunking=True.

Molecular dynamics can be run with this model, like in the run_md.py 

pot.apply_chunking = True
pot.num_chunks = 1           # total number of chunks = (num_chunk+1)^3
pot.chunk_padding = 5        # extention x (angstrom)
