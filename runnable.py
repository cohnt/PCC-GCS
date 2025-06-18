#!/usr/bin/env python
# coding: utf-8

# Restarted drake_cpy310 (Python 3.10.16)

# In[1]:


import numpy as np
from functools import partial
# from iris_plant_visualizer import IrisPlantVisualizer
import ipywidgets as widgets
from IPython.display import display
import os
import pydot


# In[2]:


# from pydrake.all import *
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
from pydrake.solvers import MosekSolver, CommonSolverOption, ScsSolver, Solve, MathematicalProgram
from pydrake.visualization import AddDefaultVisualization


# In[3]:


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

from pydrake.all import IrisParameterizationFunction


# In[4]:


from scipy.spatial import ConvexHull


# In[5]:


# import ftl_gcs


#  <h3>MultiBodyPlant Method, compatible with mimic tag right now -- 07.30</h3>

# In[6]:


meshcat = StartMeshcat()
meshcat.SetProperty("/Grid", "visible", True)
meshcat.SetProperty("/Axes", "visible", False)


# In[9]:


params = dict(
    edge_step_size=0.001
)
builder = RobotDiagramBuilder(time_step=0.001)
config_name = 'neuro_vas'
print('loading model')
builder.parser().package_map().Add("PCC-IRIS", "./assets")
directives = LoadModelDirectives(f"assets/PCC_2_5_mimic-{config_name}.yaml")
models = ProcessModelDirectives(directives, builder.plant(), builder.parser())
print('model loaded')
plant = builder.plant()
scene_graph = builder.scene_graph()

# AddDefaultVisualization(builder.Build(), meshcat)
diagram = builder.Build()

C_mapping1 = np.array([[1,0,0]])  # Mapping from joint indices to C-space indices
C_mapping2 = np.array([[0,1,0]])
C_mapping3 = np.array([[0,0,1]])
C_mapping_mat = np.concatenate((C_mapping1, 0.5*C_mapping2, C_mapping2, C_mapping2, C_mapping2, C_mapping2, C_mapping2, 0.5*C_mapping3, C_mapping3, C_mapping3, C_mapping3, C_mapping3), axis=0)


# In[10]:


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


# In[11]:


diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
scene_graph_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)


# In[12]:


diagram.ForcedPublish(diagram_context)


# In[13]:


ee_frame = plant.GetFrameByName('ee_frame')
P = ee_frame.CalcPoseInWorld(plant_context)


# In[14]:


inspector = diagram.scene_graph().model_inspector()
contacts = inspector.GetCollisionCandidates()


# In[15]:


params["robot_model_instances"] = [robot_model_instance]
params["model"] = diagram
_checker = planning.SceneGraphCollisionChecker(**params)


# In[16]:


iris_options = IrisZoOptions()
iris_options.sampled_iris_options.num_particles = 1000
iris_options.sampled_iris_options.tau = 0.5
iris_options.sampled_iris_options.delta = 1e-4
iris_options.sampled_iris_options.epsilon = 1e-4
iris_options.sampled_iris_options.max_iterations = 3
iris_options.sampled_iris_options.max_iterations_separating_planes = 500
iris_options.sampled_iris_options.max_separating_planes_per_iteration = -1
iris_options.bisection_steps = 10
iris_options.sampled_iris_options.parallelism = Parallelism(True)
iris_options.sampled_iris_options.verbose = True
iris_options.sampled_iris_options.configuration_space_margin = 1e-4
iris_options.sampled_iris_options.termination_threshold = 1e-4
iris_options.sampled_iris_options.relative_termination_threshold = 1e-3
iris_options.sampled_iris_options.random_seed = 2000
iris_options.sampled_iris_options.mixing_steps = 50
iris_options.sampled_iris_options.sample_particles_in_parallel = True
# iris_options.prog_with_additional_constraints=InverseKinematics(plant).prog()

x = Variable("x")
y = Variable("y")
z = Variable("z")
_domain = HPolyhedron.MakeBox([-0.1, -0.5,-0.5], [0.5, 0.5, 0.5])
_expression_parameterization=np.array([x,y/2, y, y, y, y, y, z/2, z, z, z, z])
_variables=np.array([x, y, z])
iris_options.parameterization = IrisParameterizationFunction(_expression_parameterization.reshape(-1,1), _variables.reshape(-1,1))

assert iris_options.parameterization.get_parameterization_is_threadsafe()

seeds=[]

# 3-shapes
# seeds.append([0.145, 0.08, 0.06])
# seeds.append([0.10, 0.06, 0.0])
# seeds.append([0.04, 0.03, 0.0])
# seeds.append([0.0, 0.0, 0.0])

# 5-shapes
# seeds.append([0.21, -0.11, 0.22])
# seeds.append([0.195, -0.1, 0.18])
# seeds.append([0.18, -0.1, 0.15])
# seeds.append([0.17, -0.08, 0.08])
# seeds.append([0.15, -0.06, 0.0])
# seeds.append([0.09, -0.03, 0.0])
# seeds.append([0.0, 0.0, 0.0])

# semi-neuro
seeds.append([0.135, -0.01, -0.155])
seeds.append([0.12, -0.01, -0.08])
seeds.append([0.08, -0.01, 0.0])
seeds.append([0.0, 0.0, 0.001])


# In[17]:


polys=[]
num_regions = 0
init_time = time.time()
for i, seed in enumerate(seeds):
    print(f"IRIS seed {i+1}: {seed}")
    # config= np.dot(C_mapping_mat, np.array(seed).reshape(-1,1))
    # iris_options.containment_points = config
    # _domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), 
    #                               plant.GetPositionUpperLimits())

    try:
        iris_options.sampled_iris_options.containment_points = np.array([[seed[0]],
                                                    [seed[1]],
                                                    [seed[2]]])
        _ellipsoid = Hyperellipsoid.MakeHypersphere(radius=0.01, center=seed)
        start_time = time.time()
        hpoly = IrisZo(_checker, _ellipsoid, _domain, iris_options)
        iris_time = time.time() - start_time
        
        print(f"✓ ({iris_time:.2f}s)")
        polys.append(hpoly)
        
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        print(f"✗ skip - {error_type}: {error_message}")
        continue  

print(f"IRIS-ZO time: {time.time()-init_time:.2f} seconds, num regions: {len(polys)}")


# In[18]:


print(polys)
print([len(poly.b()) for poly in polys])


# In[19]:


for i in range(len(polys)-1):
    if polys[i].IntersectsWith(polys[i+1]):
        print(f"IRIS region {i+1} intersects with region {i+2}")
    else:
        print(f"IRIS region {i+1} does not intersect with region {i+2}")


# In[20]:


# Code block of visualization of IRIS regions in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import ipywidgets as widgets
from IPython.display import display

def visualize_all_iris_regions_interactive(polys, seeds, elev=20, azim=45):

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    alpha_values = [0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4]
    
    print(f"processing {len(polys)} IRIS regions...") 

    fig = plt.figure(figsize=(20, 15))

    ax_main = fig.add_subplot(2, 2, (1, 3), projection='3d')

    ax_xy = fig.add_subplot(2, 2, 2)  # XY
    ax_xz = fig.add_subplot(2, 2, 4)  # XZ
    
    all_points = []
    region_data = []
    
    for region_idx, (poly, seed) in enumerate(zip(polys, seeds)):
        print(f"\n region {region_idx + 1}: seed {seed}")
        
        A = poly.A()
        b = poly.b()
        print(f"  constraint dimension: {len(b)}")
        
        search_radius = 0.08
        x_range = [seed[0] - search_radius, seed[0] + search_radius]
        y_range = [seed[1] - search_radius, seed[1] + search_radius]
        z_range = [seed[2] - search_radius, seed[2] + search_radius]
        
        resolution = 40
        x_grid = np.linspace(x_range[0], x_range[1], resolution)
        y_grid = np.linspace(y_range[0], y_range[1], resolution)
        z_grid = np.linspace(z_range[0], z_range[1], resolution)
        
        valid_points = []
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    point = np.array([x, y, z])
                    if np.all(A @ point <= b + 1e-10):
                        valid_points.append(point)
        
        valid_points = np.array(valid_points)
        
        if len(valid_points) > 0:
            print(f"   {len(valid_points)} points")
            
            x_range_actual = [valid_points[:, 0].min(), valid_points[:, 0].max()]
            y_range_actual = [valid_points[:, 1].min(), valid_points[:, 1].max()]
            z_range_actual = [valid_points[:, 2].min(), valid_points[:, 2].max()]
            
            volume = 0
            if len(valid_points) > 10:
                try:
                    hull = ConvexHull(valid_points)
                    volume = hull.volume
                    print(f"  volume: {volume:.6f}")
                except:
                    print(f"  no convex hull")
            
            region_data.append({
                'index': region_idx + 1,
                'seed': seed,
                'points': valid_points,
                'x_range': x_range_actual,
                'y_range': y_range_actual,
                'z_range': z_range_actual,
                'volume': volume,
                'point_count': len(valid_points)
            })
            
            all_points.extend(valid_points)
            
            color = colors[region_idx % len(colors)]
            alpha = alpha_values[region_idx % len(alpha_values)]
            
            ax_main.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                           alpha=0.6, s=15, c=color, 
                           label=f'Region {region_idx+1} ({len(valid_points)} pts)')
            
            ax_main.scatter(seed[0], seed[1], seed[2],
                           color=color, s=150, marker='*',
                           edgecolors='black', linewidth=2)
            
            if len(valid_points) > 10:
                try:
                    hull = ConvexHull(valid_points)
                    for simplex in hull.simplices:
                        triangle = valid_points[simplex]
                        ax_main.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                           alpha=alpha, color=color)
                except:
                    pass
            
            ax_xy.scatter(valid_points[:, 0], valid_points[:, 1], 
                         alpha=0.6, s=10, c=color, label=f'Region {region_idx+1}')
            ax_xy.scatter(seed[0], seed[1], color=color, s=100, marker='*',
                         edgecolors='black', linewidth=1)
            
            ax_xz.scatter(valid_points[:, 0], valid_points[:, 2],
                         alpha=0.6, s=10, c=color, label=f'Region {region_idx+1}')
            ax_xz.scatter(seed[0], seed[2], color=color, s=100, marker='*',
                         edgecolors='black', linewidth=1)
        else:
            print(f"  region {region_idx + 1} no valid points found")
    
    ax_main.set_xlabel('X', fontsize=12)
    ax_main.set_ylabel('Y', fontsize=12)
    ax_main.set_zlabel('Z', fontsize=12)
    ax_main.set_title('All IRIS Regions (3D Interactive)', fontsize=16)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax_main.view_init(elev=elev, azim=azim)
    
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title('XY Projection')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()
    
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('XZ Projection')
    ax_xz.grid(True, alpha=0.3)
    ax_xz.legend()
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        
        ax_main.set_xlim(x_min - 0.02, x_max + 0.02)
        ax_main.set_ylim(y_min - 0.02, y_max + 0.02)
        ax_main.set_zlim(z_min - 0.02, z_max + 0.02)
        
        ax_xy.set_xlim(x_min - 0.02, x_max + 0.02)
        ax_xy.set_ylim(y_min - 0.02, y_max + 0.02)
        
        ax_xz.set_xlim(x_min - 0.02, x_max + 0.02)
        ax_xz.set_ylim(z_min - 0.02, z_max + 0.02)
    
    ax_main.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return region_data

# region_data = visualize_all_iris_regions_interactive(polys, seeds, elev=30, azim=60)



# In[21]:


# GCS
source_config = seeds[-1]
source_q = np.dot(np.array(source_config), C_mapping_mat.transpose())


target_config = seeds[0]
target_q = np.dot(np.array(target_config), C_mapping_mat.transpose())


# In[22]:


# print(np.dot(hpoly.A(), np.array(source_config).transpose()) - hpoly.b())
print(polys[-1].PointInSet(np.array(source_config)))
print(polys[0].PointInSet(np.array(target_config)))


# In[23]:


trajopt = GcsTrajectoryOptimization(num_positions=3)
gcs = trajopt.graph_of_convex_sets()


# In[24]:


source = trajopt.AddRegions([Point(source_config)], order=2)
gcs_regions = trajopt.AddRegions(polys, order=2)
target = trajopt.AddRegions([Point(target_config)], order=2) 

ContinuityOrder = 2


# In[25]:


trajopt.AddEdges(source, gcs_regions)
trajopt.AddEdges(gcs_regions, target)
trajopt.AddPathLengthCost()
trajopt.AddPathContinuityConstraints(ContinuityOrder)
options = GraphOfConvexSetsOptions()
print('edge nums',gcs.num_edges())


# In[26]:


gcs_start_time = time.time()
[traj, result] = trajopt.SolvePath(source, target, options)
print("GCS time", time.time()-gcs_start_time)


# In[27]:


print(result.get_solution_result())
print(traj.start_time(), traj.end_time())
times = np.linspace(traj.start_time(), traj.end_time(), 1000)
waypoints = traj.vector_values(times)
import os
directory = './results/0529/2_5'
test_num = 1
if not os.path.exists(directory):
    os.makedirs(directory)
# 写入轨迹
with open(f'{directory}/{config_name}_test_{test_num}.txt', 'w') as f:
    for i in range(waypoints.shape[1]):
        f.write(' '.join([str(x) for x in waypoints[:, i]]) + '\n')

# 创建log
with open(f'{directory}/{config_name}_test_{test_num}.log', 'w') as f:
    f.write(f'config_name: {config_name}\n \n')

# 写入IRIS区域
with open(f'{directory}/{config_name}_test_{test_num}_IRIS.txt', 'w') as f:
    f.write(f'config_name: {config_name}\n \n')
    for poly in polys:
        f.write(f'poly.A' + '\n')
        f.write(f'{poly.A()}' + '\n \n')
        f.write(f'poly.B' + '\n')
        f.write(f'{poly.b()}' + '\n \n')


# In[28]:


dt = 0.05
t = 0
paths = []
time.sleep(1)
test_num = 1
print(f"Loading paths from file: {config_name}_test_{test_num}.txt")


# In[29]:


with open(f'{directory}/{config_name}_test_{test_num}.txt', 'r') as f:
# with open(f'./results/compare/0526/rrt_star_8_{config_name}.txt', 'r') as f:
    for line in f:
        q = [float(x) for x in line.split()]
        paths.append(q)
print(len(paths))
i = 0
_collision = 0
for i in range(1,len(paths)):
    config = paths[i]
    pre_config = paths[i-1]
    q = np.dot(np.array(config), C_mapping_mat.transpose())
    pre_q = np.dot(np.array(pre_config), C_mapping_mat.transpose())
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)

    q = np.array(q)
    pre_q = np.array(pre_q)
    if not _checker.CheckEdgeCollisionFree(pre_q, q):
        print("Collision occurs at step", i, "with config", q)
        _collision += 1
    if i % 100 == 0:
        print(q)
    i += 1
    # time.sleep(dt)
if not _collision:
    print("No collision occurs during the motion.")


# In[ ]:




