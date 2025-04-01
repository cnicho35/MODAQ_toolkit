# Convert multiple MCAP files
from modaq_toolkit import process_mcap_files
process_mcap_files("./test/Bag", "./test/Parquet")

# Or work with a single file
from modaq_toolkit import MCAPParser
parser = MCAPParser("./test/Bag/rosbag2_2025_04_01-15_24_10_0.mcap")
parser.read_mcap()
#parser.create_output("./Parquet")