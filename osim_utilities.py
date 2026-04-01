'''
Utilities to select and proces data from osim models.
FP 05/25
'''
import opensim
import numpy as np


def getSphereParamsFromOsimModel(model_path):
    opensim.Logger.setLevel(4) # Logger level: error
    model = opensim.Model(model_path)
    model.initSystem()

    # Get the force set and geometry set
    forceSet = model.get_ForceSet()
    geometrySet = model.get_ContactGeometrySet()

    # Loop through the force set and select only the contact spheres
    sphere_n = 0
    sphere_data = {}

    for i in range(forceSet.getSize()):        
        force = forceSet.get(i)
        if (force.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
            and force.getName()[-1] == 'r'
            ):
            force_obj = opensim.SmoothSphereHalfSpaceForce.safeDownCast(force)
        
            socket0Name = force.getSocketNames()[0]
            socket0 = force.getSocket(socket0Name)
            socket0_obj = socket0.getConnecteeAsObject()
            socket0_objName = socket0_obj.getName()            
            geo0 = geometrySet.get(socket0_objName)
            # geo0_loc = geo0.get_location().to_numpy()
            # geo0_or = geo0.get_orientation().to_numpy()
            # geo0_frameName = geo0.getFrame().getName()

            socket1Name = force.getSocketNames()[1]
            socket1 = force.getSocket(socket1Name)
            socket1_obj = socket1.getConnecteeAsObject()
            socket1_objName = socket1_obj.getName()            
            geo1 = geometrySet.get(socket1_objName)

            # Set location
            geo1_loc = geo1.get_location().to_numpy()
            # geo1_frameName = geo1.getFrame().getName()
            obj = opensim.ContactSphere.safeDownCast(geo1)

            # Set radius
            geo1_radius = obj.getRadius()

            # force_obj.set_viscous_friction(0)
            # force_obj.set_transition_velocity(0.002)

      
            # Populate a nested dictionary with sphere names and their parameters
            sphere_data[sphere_n] = {
            'loc': geo1_loc,
            'radius': geo1_radius,
            'stiffness': force_obj.get_stiffness(),
            'dissipation': force_obj.get_dissipation(),
            'static_friction': force_obj.get_static_friction(),
            'dynamic_friction': force_obj.get_dynamic_friction(),
            'viscous_friction': force_obj.get_viscous_friction(),
            'transition_velocity': force_obj.get_transition_velocity()}
            sphere_n = sphere_n + 1

    return sphere_data

def getCentersFromModel(path):
    sphere_data = getSphereParamsFromOsimModel(path)
    centers = np.zeros((len(sphere_data),3))
    for i in range(len(sphere_data)):
        centers[i,:] =  sphere_data[i]['loc']
    return centers

def getRadiiFromModel(path):
    sphere_data = getSphereParamsFromOsimModel(path)
    radii = np.zeros(len(sphere_data))
    for i in range(len(sphere_data)):
        radii[i] =  sphere_data[i]['radius']
    return radii
