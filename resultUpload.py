import roboflow
# b5d7fffa-2d17-47b9-81c6-c4270a1ad64a
rf = roboflow.Roboflow('R9lecSMIgT9APWv8zuzD')

workspace = rf.workspace("npust-sym1f")
project = workspace.project("twroad")
version = project.version(1)


