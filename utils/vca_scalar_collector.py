


class VcaScalarCollector(object):
    def __init__(self) -> None:
        self.data = {}
        pass

    def add_scalar(self, path,scalar):
        if path in self.data:
            self.data[path].append(scalar)
        else:
            self.data[path] = [scalar]

    def get_scalar(self,path):
        if path in self.data:
            if len(self.data[path]) > 1:
                return sum(self.data[path])/len(self.data[path])
            return self.data[path]
        else:
            return 0

    def publish(self,function,iteration:int):
        # Summarize
        summarized = {}
        grouped_keys = {}
        for key in self.data.keys():
            parts = key.split("/")
            # Extract the first part and the last two parts
            group_name = parts[0]
            sub_key = "/".join(parts[-2:])

            # If the sub key does not exist in the dictionary, create a new entry with an empty dictionary
            if sub_key not in grouped_keys:
                grouped_keys[sub_key] = {}

            # Assign the group_name to the dictionary under the sub_key
            if group_name not in grouped_keys[sub_key]:
                grouped_keys[sub_key][group_name] = sum(self.data[key])/len(self.data[key])

            # Append an element to the list under the corresponding group name
            # grouped_keys[sub_key][group_name].append(sum(self.data[key])/len(self.data[key]))
        
        for key in grouped_keys:
            function(key,grouped_keys[key],iteration)
        # Clear
        self.data.clear()
        

