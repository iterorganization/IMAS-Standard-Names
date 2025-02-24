import strictyaml

with open ("../standardnames.yml", "r") as f:
    standard_names = strictyaml.load(f.read())

print(standard_names)