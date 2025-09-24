import main  # Import our machine info module
import time 
def test_dict_import():
    """Test importing machine info as a dictionary"""
    print("=== Testing dictionary import ===")
    machine_info = main.get_machine_info_dict()
    print(f"Type: {type(machine_info)}")
    print(f"Keys: {list(machine_info.keys())}")
    print(f"Hostname: {machine_info.get('hostname')}")
    print(f"CPU count: {machine_info.get('number_of_cpu')}")
    print()

def test_json_import():
    """Test importing machine info as JSON string"""
    print("=== Testing JSON string import ===")
    machine_json = main.get_machine_info_json()
    print(f"Type: {type(machine_json)}")
    print("First 200 characters of JSON:")
    print(machine_json)
    print()

def test_json_parsing():
    """Test that the JSON can be parsed back to dict"""
    print("=== Testing JSON parsing ===")
    import json
    machine_json = main.get_machine_info_json()
    parsed_dict = json.loads(machine_json)
    print(f"Successfully parsed JSON back to dict: {type(parsed_dict)}")
    print(f"Username from parsed JSON: {parsed_dict.get('username')}")
    print()

if __name__ == "__main__":
    print("Testing machine info module import capabilities...\n")
    # test_dict_import()
    start_time = time.time()
    test_json_import() 
    # test_json_parsing()
    print("All tests completed successfully!")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
