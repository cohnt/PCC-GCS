import numpy as np
from functools import partial
import ipywidgets as widgets
from IPython.display import display
import os

from pydrake.common import Parallelism
from pydrake.systems.analysis import Simulator
import time
from pydrake.geometry import (MeshcatVisualizer, 
                              MeshcatVisualizerParams, 
                              Role, 
                              StartMeshcat, 
                              GeometryInstance,
                              MakePhongIllustrationProperties,
                              SceneGraph,
                              Sphere, Box)
# from pydrake.geometry.optimization import IrisOptions, IrisInRationalConfigurationSpace, HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization import (IrisOptions, 
                                           IrisInConfigurationSpace, 
                                           HPolyhedron, 
                                           Hyperellipsoid, 
                                           GraphOfConvexSetsOptions,
                                           Point)
from pydrake.visualization import AddDefaultVisualization

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.tree import RevoluteJoint, PrismaticJoint
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives, AddCollisionFilterGroup
from pydrake.multibody.plant import (AddMultibodyPlantSceneGraph,
                                     MultibodyPlant, 
                                     DiscreteContactSolver,
                                     DiscreteContactApproximation)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import RigidTransform
from pydrake.planning import IrisZo,IrisZoOptions, GcsTrajectoryOptimization, RobotDiagramBuilder
import pydrake.planning as planning
from pydrake.symbolic import Variable

# %%
from scipy.spatial import ConvexHull

meshcat = StartMeshcat()
meshcat.SetProperty("/Grid", "visible", True)
meshcat.SetProperty("/Axes", "visible", False)

# %%
params = dict(
    edge_step_size=0.001
)
builder = RobotDiagramBuilder(time_step=0.001)
config_name = ''
builder.parser().package_map().Add("", "./assets")
directives = LoadModelDirectives(f"assets/{config_name}.yaml")
models = ProcessModelDirectives(directives, builder.plant(), builder.parser())

plant = builder.plant()
scene_graph = builder.scene_graph()

# AddDefaultVisualization(builder.Build(), meshcat)
diagram = builder.Build()

C_mapping1 = np.array([[1,0,0]])  # Mapping from joint indices to C-space indices
C_mapping2 = np.array([[0,1,0]])
C_mapping3 = np.array([[0,0,1]])
C_mapping_mat = np.concatenate((C_mapping1, 0.5*C_mapping2, C_mapping2, C_mapping2, C_mapping2, C_mapping2, C_mapping2, 0.5*C_mapping3, C_mapping3, C_mapping3, C_mapping3, C_mapping3), axis=0)

q0 = []

robot_model_instance = plant.GetModelInstanceByName("PCC")
index = 0
for joint_index in plant.GetJointIndices(robot_model_instance):
    joint = plant.get_mutable_joint(joint_index)
    print(joint)
    if isinstance(joint, RevoluteJoint):
        q0.append(0.0)
        joint.set_default_angle(np.array(q0[index]))
    elif isinstance(joint, PrismaticJoint):
        q0.append(0.0)
        joint.set_default_translation(np.array(q0[index]))
    index += 1

# %%
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
scene_graph_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)

# %%
diagram.ForcedPublish(diagram_context)

# %%
ee_frame = plant.GetFrameByName('ee_frame')
P = ee_frame.CalcPoseInWorld(plant_context)

# %%
inspector = diagram.scene_graph().model_inspector()
contacts = inspector.GetCollisionCandidates()

# %% IRIS-ZO
params["robot_model_instances"] = [robot_model_instance]
params["model"] = diagram
_checker = planning.SceneGraphCollisionChecker(**params)

iris_options = IrisZoOptions()
iris_options.num_particles = 10000
iris_options.tau = 0.3
iris_options.delta = 5e-1
iris_options.epsilon = 1e-2
iris_options.max_iterations = 50
iris_options.max_iterations_separating_planes = 50
iris_options.max_separating_planes_per_iteration = -1
iris_options.bisection_steps = 10
iris_options.parallelism = Parallelism(True)
iris_options.verbose = True
iris_options.configuration_space_margin = 1e-4
iris_options.termination_threshold = 1e-4
iris_options.relative_termination_threshold = 1e-3
iris_options.random_seed = 2000
iris_options.mixing_steps = 50
# iris_options.prog_with_additional_constraints=InverseKinematics(plant).prog()

x = Variable("x")
y = Variable("y")
z = Variable("z")
_domain = HPolyhedron.MakeBox([-0.1, -0.5,-0.5], [0.5, 0.5, 0.5])
_expression_parameterization=np.array([x,y/2, y, y, y, y, y, z/2, z, z, z, z])
_variables=np.array([x, y, z])
iris_options.SetParameterizationFromExpression(_expression_parameterization.reshape(-1,1), _variables.reshape(-1,1))

seeds=[]

seeds.append([0.135, -0.01, -0.16])
seeds.append([0.12, -0.01, -0.08])
seeds.append([0.08, -0.01, 0.0])
seeds.append([0.0, 0.0, 0.0])

polys=[]
for i, seed in enumerate(seeds):
    print(f"IRIS seed {i+1}: {seed}")
    iris_options.containment_points = np.array([[seed[0]],
                                                [seed[1]],
                                                [seed[2]]])
    _ellipsoid = Hyperellipsoid.MakeHypersphere(radius=0.01, center=seed)
    start_time = time.time()
    hpoly = IrisZo(_checker, _ellipsoid, _domain, iris_options)
    print(f"IRIS time for seed {i+1}", time.time()-start_time)
    polys.append(hpoly)

# %%
for i in range(len(polys)-1):
    if polys[i].IntersectsWith(polys[i+1]):
        print(f"IRIS region {i+1} intersects with region {i+2}")
    else:
        print(f"IRIS region {i+1} does not intersect with region {i+2}")


# GCS
source_config = seeds[-1]
source_q = np.dot(np.array(source_config), C_mapping_mat.transpose())

target_config = seeds[0]
target_q = np.dot(np.array(target_config), C_mapping_mat.transpose())

trajopt = GcsTrajectoryOptimization(num_positions=3)
gcs = trajopt.graph_of_convex_sets()

source = trajopt.AddRegions([Point(source_config)], order=2)
gcs_regions = trajopt.AddRegions(polys, order=2)
target = trajopt.AddRegions([Point(target_config)], order=2) 

ContinuityOrder = 2

# %% 
trajopt.AddEdges(source, gcs_regions)
trajopt.AddEdges(gcs_regions, target)
trajopt.AddPathLengthCost()
trajopt.AddPathContinuityConstraints(ContinuityOrder)
options = GraphOfConvexSetsOptions()
print('edge nums',gcs.num_edges())
#%% 执行GCS
gcs_start_time = time.time()
[traj, result] = trajopt.SolvePath(source, target, options)
print("GCS time", time.time()-gcs_start_time)
