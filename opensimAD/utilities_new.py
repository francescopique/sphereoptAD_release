import os
import sys
import opensim
import numpy as np
import casadi as ca
import shutil
import importlib
import pandas as pd
import platform
import urllib.request
import zipfile
from opensim_sto_reader import readMotionFile

def generateExternalFunction(pathOpenSimModel, outputDir, pathID,
                             outputFilename='F'):
    
    # %% Paths.
    os.makedirs(outputDir, exist_ok=True)
    pathOutputFile = os.path.join(outputDir, outputFilename + ".cpp")
    pathOutputMap = os.path.join(outputDir, outputFilename + "_map.npy")
    
    # %% Generate external Function (.cpp file).
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathOpenSimModel)
    model.initSystem()
    bodySet = model.getBodySet()
    
    nBodies = 0
    for i in range(bodySet.getSize()):        
        c_body = bodySet.get(i)
        c_body_name = c_body.getName()  
        if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
            continue
        nBodies += 1
    
    jointSet = model.get_JointSet()
    nJoints = jointSet.getSize()
    geometrySet = model.get_ContactGeometrySet()
    forceSet = model.get_ForceSet()
    coordinateSet = model.getCoordinateSet()
    nCoordinates = coordinateSet.getSize()
    coordinates = []
    for coor in range(nCoordinates):
        coordinates.append(coordinateSet.get(coor).getName())
    sides = ['r', 'l']
    for side in sides:
        # We do not include the coordinates from the patellofemoral joints,
        # since they only influence muscle paths, which we approximate using
        # polynomials offline.
        if 'knee_angle_{}_beta'.format(side) in coordinates:
            nCoordinates -= 1
            nJoints -= 1
        elif 'knee_angle_beta_{}'.format(side) in coordinates:
            nCoordinates -= 1
            nJoints -= 1        
    
    nContacts = 0
    for i in range(forceSet.getSize()):        
        c_force_elt = forceSet.get(i)        
        if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":  
            nContacts += 1
    
    with open(pathOutputFile, "w") as f:
        
        # TODO: only include those that are necessary (model-specific).
        f.write('#include <OpenSim/Simulation/Model/Model.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/PlanarJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/Joint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>\n')
        f.write('#include <OpenSim/Common/LinearFunction.h>\n')
        f.write('#include <OpenSim/Common/PolynomialFunction.h>\n')
        f.write('#include <OpenSim/Common/MultiplierFunction.h>\n')
        f.write('#include <OpenSim/Common/Constant.h>\n')
        f.write('#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>\n')
        f.write('#include <OpenSim/Simulation/SimulationUtilities.h>\n')
        f.write('#include "SimTKcommon/internal/recorder.h"\n\n')
        
        f.write('#include <iostream>\n')
        f.write('#include <iterator>\n')
        f.write('#include <random>\n')
        f.write('#include <cassert>\n')
        f.write('#include <algorithm>\n')
        f.write('#include <vector>\n')
        f.write('#include <fstream>\n\n')
        
        f.write('using namespace SimTK;\n')
        f.write('using namespace OpenSim;\n\n')
    
        f.write('constexpr int n_in = 4; \n')
        f.write('constexpr int n_out = 1; \n')
        
        f.write('constexpr int nCoordinates = %i; \n' % nCoordinates)

        # X : stati, U: controlli
        # f.write('constexpr int NX = nCoordinates*2; \n') # No intertwining of qs and?
        f.write('constexpr int NX = 2*nCoordinates; \n')

        # f.write('constexpr int NU = nCoordinates; \n')
        
        # 3 GRFs, 1 My, and 2 COP
        nOutputs = 6
        f.write('constexpr int NR = %i; \n\n' % (nOutputs))

        # helper function T
        f.write('template<typename T> \n')
        f.write('T value(const Recorder& e) { return e; }; \n')
        f.write('template<> \n')
        f.write('double value(const Recorder& e) { return e.getValue(); }; \n\n')
        
        f.write('template<typename T>\n')
        f.write('int F_generic(const T** arg, T** res) {\n\n')
        
        # Model
        f.write('\t// Definition of model.\n')
        f.write('\tOpenSim::Model* model;\n')
        f.write('\tmodel = new OpenSim::Model();\n\n')
        
        # Bodies
        f.write('\t// Definition of bodies.\n')
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue            
            c_body_mass = c_body.get_mass()
            c_body_mass_center = c_body.get_mass_center().to_numpy()
            c_body_inertia = c_body.get_inertia()
            c_body_inertia_vec3 = np.array([c_body_inertia.get(0), c_body_inertia.get(1), c_body_inertia.get(2)])        
            f.write('\tOpenSim::Body* %s;\n' % c_body_name)
            f.write('\t%s = new OpenSim::Body(\"%s\", %.20f, Vec3(%.20f, %.20f, %.20f), Inertia(%.20f, %.20f, %.20f, 0., 0., 0.));\n' % (c_body_name, c_body_name, c_body_mass, c_body_mass_center[0], c_body_mass_center[1], c_body_mass_center[2], c_body_inertia_vec3[0], c_body_inertia_vec3[1], c_body_inertia_vec3[2]))
            f.write('\tmodel->addBody(%s);\n' % (c_body_name))
            f.write('\n')
        
        # Joints
        f.write('\t// Definition of joints.\n')
        for i in range(jointSet.getSize()): 
            c_joint = jointSet.get(i)
            c_joint_type = c_joint.getConcreteClassName()
            
            c_joint_name = c_joint.getName()
            if (c_joint_name == 'patellofemoral_l' or 
                c_joint_name == 'patellofemoral_r'):
                continue
            
            parent_frame = c_joint.get_frames(0)
            parent_frame_name = parent_frame.getParentFrame().getName()
            parent_frame_trans = parent_frame.get_translation().to_numpy()
            parent_frame_or = parent_frame.get_orientation().to_numpy()
            
            child_frame = c_joint.get_frames(1)
            child_frame_name = child_frame.getParentFrame().getName()
            child_frame_trans = child_frame.get_translation().to_numpy()
            child_frame_or = child_frame.get_orientation().to_numpy()
            
            # Custom joints
            if c_joint_type == "CustomJoint":

                f.write('\tSpatialTransform st_%s;\n' % c_joint.getName())

                cObj = opensim.CustomJoint.safeDownCast(c_joint)
                spatialtransform = cObj.get_SpatialTransform()

                for iCoord in range(6):
                    if iCoord == 0:
                        dofSel = spatialtransform.get_rotation1()
                    elif iCoord == 1:
                        dofSel = spatialtransform.get_rotation2()
                    elif iCoord == 2:
                        dofSel = spatialtransform.get_rotation3()
                    elif iCoord == 3:
                        dofSel = spatialtransform.get_translation1()
                    elif iCoord == 4:
                        dofSel = spatialtransform.get_translation2()
                    elif iCoord == 5:
                        dofSel = spatialtransform.get_translation3()
                    coord = iCoord

                    # Transform axis.
                    dofSel_axis = dofSel.get_axis().to_numpy()
                    dofSel_f = dofSel.get_function()
                    if dofSel_f.getConcreteClassName() == 'LinearFunction':
                        dofSel_f_obj = opensim.LinearFunction.safeDownCast(dofSel_f)
                        dofSel_f_slope = dofSel_f_obj.getSlope()
                        dofSel_f_intercept = dofSel_f_obj.getIntercept()
                        #c_coord = c_joint.get_coordinates(coord)
                        c_coord_name = dofSel.get_coordinates(0)
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                        c_joint.getName(), coord, c_coord_name))
                        f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (
                        c_joint.getName(), coord, dofSel_f_slope, dofSel_f_intercept))
                    elif dofSel_f.getConcreteClassName() == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                        c_joint.getName(), coord, c_coord_name))
                        dofSel_f_obj = opensim.PolynomialFunction.safeDownCast(dofSel_f)
                        dofSel_f_coeffs = dofSel_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = dofSel_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                            dofSel_f_coeffs[3]))
                        elif c_nCoeffs == 5:
                            f.write(
                                '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                                dofSel_f_coeffs[3], dofSel_f_coeffs[4]))
                        elif c_nCoeffs == 7:
                            f.write(
                                '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                                dofSel_f_coeffs[3], dofSel_f_coeffs[4], dofSel_f_coeffs[5], dofSel_f_coeffs[6]))
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (
                        c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (
                        c_joint.getName(), coord, c_joint.getName(), coord))
                    elif dofSel_f.getConcreteClassName() == 'MultiplierFunction':
                        dofSel_f_obj = opensim.MultiplierFunction.safeDownCast(dofSel_f)
                        dofSel_f_obj_scale = dofSel_f_obj.getScale()
                        dofSel_f_obj_f = dofSel_f_obj.getFunction()
                        dofSel_f_obj_f_name = dofSel_f_obj_f.getConcreteClassName()
                        if dofSel_f_obj_f_name == 'Constant':
                            dofSel_f_obj_f_obj = opensim.Constant.safeDownCast(dofSel_f_obj_f)
                            dofSel_f_obj_f_obj_value = dofSel_f_obj_f_obj.getValue()
                            f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (
                            c_joint.getName(), coord, dofSel_f_obj_f_obj_value, dofSel_f_obj_scale))
                        elif dofSel_f_obj_f_name == 'PolynomialFunction':
                            f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                            c_joint.getName(), coord, c_coord_name))
                            dofSel_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(dofSel_f_obj_f)
                            dofSel_f_obj_f_coeffs = dofSel_f_obj_f_obj.getCoefficients().to_numpy()
                            c_nCoeffs = dofSel_f_obj_f_coeffs.shape[0]
                            if c_nCoeffs == 2:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1]))
                            elif c_nCoeffs == 3:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                dofSel_f_obj_f_coeffs[2]))
                            elif c_nCoeffs == 4:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                dofSel_f_obj_f_coeffs[2], dofSel_f_obj_f_coeffs[3]))
                            elif c_nCoeffs == 5:
                                f.write(
                                    '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                    c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                    dofSel_f_obj_f_coeffs[2], dofSel_f_obj_f_coeffs[3], dofSel_f_obj_f_coeffs[4]))
                            else:
                                raise ValueError("TODO")
                            f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                            f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (
                            c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                            f.write(
                                '\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (
                                c_joint.getName(), coord, c_joint.getName(), coord, dofSel_f_obj_scale))
                        else:
                            raise ValueError("Not supported")
                    elif dofSel_f.getConcreteClassName() == 'Constant':
                        dofSel_f_obj = opensim.Constant.safeDownCast(dofSel_f)
                        dofSel_f_obj_value = dofSel_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (
                        c_joint.getName(), coord, dofSel_f_obj_value))
                    else:
                        raise ValueError(dofSel_f.getConcreteClassName() +" Not supported")
                    f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (
                    c_joint.getName(), coord, dofSel_axis[0], dofSel_axis[1], dofSel_axis[2]))
                
                
                # Joint.
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))
                
            elif c_joint_type == 'PinJoint' or c_joint_type == 'WeldJoint' or c_joint_type == 'PlanarJoint':
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2]))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2])) 
            else:
                raise ValueError("TODO: joint type not yet supported")
            f.write('\tmodel->addJoint(%s);\n' % (c_joint.getName()))
            f.write('\n')  
        


        # dimensions of this must be adjusted, also in main, according to our problem
        # think about how th edll will be usde in the casadi main
 
        # We probably don't care about u (controls)
        f.write('\t//read inputs for spheres\n')
        f.write('\tT E(arg[1][0]);\n')
        f.write('\tT c(arg[2][0]);\n')
        f.write('\tT static_friction = arg[3][0];\n\n')


        
        # f.write('\t// States and controls.\n')
        # f.write('\tT ua[NU];\n')
        # f.write('\t/// OpenSim and Simbody have different state orders.\n')
        # f.write('\tauto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);\n')
        # do we care about these?
        # f.write('\tfor (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];\n\n')

        # Contacts
        f.write('\t// Definition of contacts.\n')
        sphere_n = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)
            
            if (c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
                and c_force_elt.getName()[-1] == 'r'
                ):            
                c_force_elt_obj =  opensim.SmoothSphereHalfSpaceForce.safeDownCast(c_force_elt) 	
                
                socket0Name = c_force_elt.getSocketNames()[0]
                socket0 = c_force_elt.getSocket(socket0Name)
                socket0_obj = socket0.getConnecteeAsObject()
                socket0_objName = socket0_obj.getName()            
                geo0 = geometrySet.get(socket0_objName)
                geo0_loc = geo0.get_location().to_numpy()
                geo0_or = geo0.get_orientation().to_numpy()
                geo0_frameName = geo0.getFrame().getName()
                
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_loc = geo1.get_location().to_numpy()
                # geo1_or = geo1.get_orientation().to_numpy()
                geo1_frameName = geo1.getFrame().getName()
                obj = opensim.ContactSphere.safeDownCast(geo1) 	
                geo1_radius = obj.getRadius()            
                
                f.write('\tOpenSim::%s* %s;\n' % (c_force_elt.getConcreteClassName(), c_force_elt.getName()))
                if geo0_frameName == "ground":
                    f.write('\t%s = new %s(\"%s\", *%s, model->getGround());\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName))
                else:
                    f.write('\t%s = new %s(\"%s\", *%s, *%s);\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName, geo0_frameName))
                
                # ADD HERE OUR INPUTS: (probably we will anticipate the variables definition)

                f.write('\tVec3 %s_location(%.20f, %.20f, %.20f);\n' % (c_force_elt.getName(), geo1_loc[0], geo1_loc[1], geo1_loc[2]))
                f.write('\t%s->set_contact_sphere_location(%s_location);\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\tdouble %s_radius = (%.20f);\n' % (c_force_elt.getName(), geo1_radius))
                f.write('\t%s->set_contact_sphere_radius(%s_radius );\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\t%s->set_contact_half_space_location(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_loc[0], geo0_loc[1], geo0_loc[2]))
                f.write('\t%s->set_contact_half_space_orientation(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_or[0], geo0_or[1], geo0_or[2]))
                
                f.write('\t%s->set_stiffness(E);\n' % (c_force_elt.getName()))
                f.write('\t%s->set_dissipation(c);\n' % (c_force_elt.getName()))
                f.write('\t%s->set_static_friction(static_friction);\n' % (c_force_elt.getName()))

                # Old friction and trans velocity settings:
                # f.write('\t%s->set_dynamic_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_dynamic_friction()))
                # f.write('\t%s->set_viscous_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_viscous_friction()))
                # f.write('\t%s->set_transition_velocity(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_transition_velocity()))
                
                # Here we hard code the dynamic and viscous friction and the transition velocity:
                f.write('\t%s->set_dynamic_friction(static_friction);\n' % (c_force_elt.getName()))
                f.write('\t%s->set_viscous_friction(0);\n' % (c_force_elt.getName()))
                f.write('\t%s->set_transition_velocity(0.065);\n' % (c_force_elt.getName()))


                f.write('\t%s->connectSocket_sphere_frame(*%s);\n' % (c_force_elt.getName(), geo1_frameName))
                if geo0_frameName == "ground":
                    f.write('\t%s->connectSocket_half_space_frame(model->getGround());\n' % (c_force_elt.getName()))                
                else:
                    f.write('\t%s->connectSocket_half_space_frame(*%s);\n' % (c_force_elt.getName(), geo0_frameName))
                f.write('\tmodel->addComponent(%s);\n' % (c_force_elt.getName()))
                f.write('\n')
                sphere_n += 1
            
        f.write('\t// Initialize system.\n')
        f.write('\tSimTK::State* state;\n')
        f.write('\tstate = new State(model->initSystem());\n\n')

        f.write('\t// Read input coords.\n')
        f.write('\tstd::vector<T> x(arg[0], arg[0] + NX);\n')
        f.write('\tVector QsUs(NX);\n')
        f.write('\tfor (int i = 0; i < NX; ++i) QsUs[i] = x[i];\n') 	
        # f.write('\t/// Controls\n')

        f.write('\t// Set state variables and realize.\n')
        f.write('\tmodel->setStateVariableValues(*state, QsUs);\n')
        f.write('\tmodel->realizeVelocity(*state);\n\n')
        
        # f.write('\t// Compute residual forces.\n')
        # f.write('\t/// Set appliedMobilityForces (# mobilities).\n')
        # f.write('\tVector appliedMobilityForces(nCoordinates);\n')
        # f.write('\tappliedMobilityForces.setToZero();\n')
        # f.write('\t/// Set appliedBodyForces (# bodies + ground).\n')
        # f.write('\tVector_<SpatialVec> appliedBodyForces;\n')
        # f.write('\tint nbodies = model->getBodySet().getSize() + 1;\n')
        # f.write('\tappliedBodyForces.resize(nbodies);\n')
        # f.write('\tappliedBodyForces.setToZero();\n')
        # f.write('\t/// Set gravity.\n')
        # f.write('\tVec3 gravity(0);\n')
        # f.write('\tgravity[1] = %.20f;\n' % model.get_gravity()[1])
        # f.write('\t/// Add weights to appliedBodyForces.\n')
        # f.write('\tfor (int i = 0; i < model->getBodySet().getSize(); ++i) {\n')
        # f.write('\t\tmodel->getMatterSubsystem().addInStationForce(*state,\n')
        # f.write('\t\tmodel->getBodySet().get(i).getMobilizedBodyIndex(),\n')
        # f.write('\t\tmodel->getBodySet().get(i).getMassCenter(),\n')
        # f.write('\t\tmodel->getBodySet().get(i).getMass()*gravity, appliedBodyForces);\n')
        # f.write('\t}\n')    
        # f.write('\t/// Add contact forces to appliedBodyForces.\n')
        
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)     
            
            if (c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
                and c_force_elt.getName()[-1] == 'r'
                ):
                c_force_elt_name = c_force_elt.getName()    
                
                f.write('\tArray<osim_double_adouble> Force_%s = %s->getRecordValues(*state);\n' % (str(count), c_force_elt_name))
                f.write('\tSpatialVec GRF_%s;\n' % (str(count)))           
                
                f.write('\tGRF_%s[0] = Vec3(Force_%s[3], Force_%s[4], Force_%s[5]);\n' % (str(count), str(count), str(count), str(count)))
                f.write('\tGRF_%s[1] = Vec3(Force_%s[0], Force_%s[1], Force_%s[2]);\n' % (str(count), str(count), str(count), str(count)))
                
                # socket1Name = c_force_elt.getSocketNames()[1]
                # socket1 = c_force_elt.getSocket(socket1Name)
                # socket1_obj = socket1.getConnecteeAsObject()
                # socket1_objName = socket1_obj.getName()            
                # geo1 = geometrySet.get(socket1_objName)
                # geo1_frameName = geo1.getFrame().getName()
                
                # f.write('\tint c_idx_%s = model->getBodySet().get("%s").getMobilizedBodyIndex();\n' % (str(count), geo1_frameName))            
                # f.write('\tappliedBodyForces[c_idx_%s] += GRF_%s;\n' % (str(count), str(count)))
                count += 1
                f.write('\n')
                
        # f.write('\t/// knownUdot.\n')
        # f.write('\tVector knownUdot(nCoordinates);\n')
        # f.write('\tknownUdot.setToZero();\n')
        # f.write('\tfor (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];\n')
        # f.write('\t/// Calculate residual forces.\n')
        # f.write('\tVector residualMobilityForces(nCoordinates);\n')
        # f.write('\tresidualMobilityForces.setToZero();\n')
        # f.write('\tmodel->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,\n')
        # f.write('\t\t\tappliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);\n\n')
        
        # Get body origins.
        # f.write('\t/// Body origins.\n')
        # for i in range(bodySet.getSize()):        
        #     c_body = bodySet.get(i)
        #     c_body_name = c_body.getName()            
        #     if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
        #         continue            
        #     f.write('\tVec3 %s_or = %s->getPositionInGround(*state);\n' % (c_body_name, c_body_name))
        # f.write('\n')
            
        # Get GRFs.
        f.write('\t/// Ground reaction forces.\n')
        f.write('\tVec3 GRF_r(0), GRF_l(0);\n')
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if (c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
                and c_force_elt.getName()[-1] == 'r'
                ):
                c_force_elt_name = c_force_elt.getName() 
                # if c_force_elt_name[-2:] == "_r":
                f.write('\tGRF_r += GRF_%s[1];\n'  % (str(count)))
                # elif c_force_elt_name[-2:] == "_l":
                #     # f.write('\tGRF_l += GRF_%s[1];\n'  % (str(count)))
                # else:
                #     raise ValueError("Cannot identify contact side")
                count += 1
        f.write('\n')
            
        # Get GRMs.
        f.write('\t/// Ground reaction moments.\n')
        f.write('\tVec3 GRM_r(0), GRM_l(0), MHOR(0);\n')
        f.write('\tVec3 normal(0, 1, 0);\n\n')
        count = 0
        geo1_frameNames = []
        # f.write('\tVec3 tmp(0);\n')
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if (c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
                and c_force_elt.getName()[-1] == 'r'
                ):
                c_force_elt_name = c_force_elt.getName() 
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName() 
                
                if not geo1_frameName in geo1_frameNames:
                    f.write('\tSimTK::Transform TR_GB_%s = %s->getMobilizedBody().getBodyTransform(*state);\n' % (geo1_frameName, geo1_frameName))    
                    geo1_frameNames.append(geo1_frameName)
                    
                f.write('\tVec3 %s_location_G = %s->findStationLocationInGround(*state, %s_location);\n' % (c_force_elt_name, geo1_frameName, c_force_elt_name))                
                f.write('\tVec3 %s_locationCP_G = %s_location_G - %s_radius * normal;\n' % (c_force_elt_name, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 locationCP_G_adj_%i = %s_locationCP_G - 0.5*%s_locationCP_G[1] * normal;\n' % (count, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 %s_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_%i, *%s);\n' % (c_force_elt_name, count, geo1_frameName))
                f.write('\tVec3 GRM_%i = (TR_GB_%s*%s_locationCP_B) %% GRF_%s[1];\n' % (count, geo1_frameName, c_force_elt_name, str(count)))
                # if c_force_elt_name[-2:] == "_r":
                f.write('\tGRM_r += GRM_%i;\n'  % (count))
                # f.write('\tfor (int i = 0; i < 3; ++i) tmp[i] += %s_locationCP_G[i]*GRF_%s[1][1];\n' % (c_force_elt_name, str(count)))
                    # f.write('\ttmp += %s_locationCP_G * GRF_%s[1][1];\n' % (c_force_elt_name, str(count)))
   
                # elif c_force_elt_name[-2:] == "_l": 
                #     f.write('\tGRM_l += GRM_%i;\n'  % (count))   
                # else:
                #     raise ValueError("Cannot identify contact side")
                f.write('\n')                   
                count += 1
        # f.write('\tVec3 COP(0);\n')
        # f.write('\tfor(int i = 0; i < 3; ++i) COP[i] = tmp[i] / GRF_r[1];\n')
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if (c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
                and c_force_elt.getName()[-1] == 'r'
                ):
                c_force_elt_name = c_force_elt.getName() 
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName() 
                # f.write('\tMHOR += (%s_locationCP_G - COP) %% GRF_%s[1];\n' % (c_force_elt_name, str(count)))
                count += 1
                
        # Save dict pointing to which elements are returned by F and in which
        # order, such as to facilitate using F when formulating problem.
        # F_map = {}
        
        # f.write('\t/// Outputs.\n')        
        # # Export residuals (joint torques).
        # f.write('\t/// Residual forces (OpenSim and Simbody have different state orders).\n')
        # f.write('\tauto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);\n')
        # f.write('\tfor (int i = 0; i < NU; ++i) res[0][i] =\n')
        # f.write('\t\t\tvalue<T>(residualMobilityForces[indicesSimbodyInOS[i]]);\n')
        # F_map['residuals'] = {}
        # count = 0
        # for coordinate in coordinates:
        #     if 'beta' in coordinate:
        #         continue
        #     F_map['residuals'][coordinate] = count 
        #     count += 1
        # count_acc = nCoordinates
        
        # Export GRFs
        f.write('\t/// Ground reaction forces.\n')
        f.write('\tfor (int i = 0; i < 3; ++i) res[0][i] = value<T>(GRF_r[i]);\n')
        # f.write('\tfor (int i = 3; i < 6; ++i) res[0][i] = value<T>(GRM_r[i]);\n')
        # F_map['GRFs'] = {}      
        # F_map['GRFs']['right'] = range(count_acc, count_acc+3)
        # F_map['GRFs']['left'] = range(count_acc+3, count_acc+6)
        # # count_acc += 6

        
        # f.write('\tT COPx = (GRM_r[2] - 0.005*GRF_r[0]) / GRF_r[1];\n')
        # f.write('\tT COPz = (-GRM_r[0] - 0.005*GRF_r[2]) / GRF_r[1];\n') 
        # f.write('\tT My = GRM_r[1] - GRF_r[0]*COPz + GRF_r[2]*COPx ;\n')
 

        f.write('\tT COPx = GRM_r[2] / GRF_r[1];\n')
        f.write('\tT COPz = -GRM_r[0] / GRF_r[1];\n') 
        f.write('\tT My = GRM_r[1] - GRF_r[0]*COPz + GRF_r[2]*COPx ;\n')

        # Export GRMs
        # f.write('\t/// Ground reaction moments e COP.\n')
        f.write('\tres[0][3] = value<T>(COPx);\n')
        f.write('\tres[0][4] = value<T>(My);\n')
        f.write('\tres[0][5] = value<T>(COPz);\n')
        # f.write('\tres[0][6] = value<T>(COP[1]);\n')



        f.write('\n')
        f.write('\treturn 0;\n')
        f.write('}\n\n')
        
        f.write('int main() {\n')
        f.write('\tRecorder E[1];\n')
        f.write('\tRecorder c[1];\n')
        f.write('\tRecorder f[1];\n')
        f.write('\tRecorder x[NX];\n')

        f.write('\tRecorder GRF[NR];\n')
        f.write('\tfor (int i = 0; i < NX; ++i) x[i] <<= 0;\n')
        f.write('\tfor (int i = 0; i < 1; ++i) E[i] <<= 0;\n')
        f.write('\tfor (int i = 0; i < 1; ++i) c[i] <<= 0;\n')
        f.write('\tfor (int i = 0; i < 1; ++i) f[i] <<= 0;\n')

        f.write('\tconst Recorder* Recorder_arg[n_in] = { x,E,c, f };\n')
        f.write('\tRecorder* Recorder_res[n_out] = { GRF };\n')
        f.write('\tF_generic<Recorder>(Recorder_arg, Recorder_res);\n')
        f.write('\tdouble res[NR];\n')
        f.write('\tfor (int i = 0; i < NR; ++i) Recorder_res[0][i] >>= res[i];\n')
        f.write('\tRecorder::stop_recording();\n')
        f.write('\treturn 0;\n')
        f.write('}\n')
        
        # Save dict
        # np.save(pathOutputMap, F_map)
        
    # %% Build external Function.
    # 2*ncoords + nspheres * 6 + 1(friction)
    buildExternalFunction(outputFilename, outputDir, 2*nCoordinates+3)
    print("Generated external function!")
    
            
    # # %% GRF verification test.
    # # Delete previous saved dummy motion if needed.
    # if os.path.exists(os.path.join(pathID, "dummyData.sto")):
    #     os.remove(os.path.join(pathID, "dummyData.sto"))

    # # # Create a dummy motion for ID.
    # nCoordinatesAll = coordinateSet.getSize()
    # dummyDataa = np.zeros((10, nCoordinatesAll + 1))
    # for coor in range(nCoordinatesAll):
    #     dummyDataa[:, coor + 1] = np.random.rand()*0.05
    # dummyDataa[:, 0] = np.linspace(0.01, 0.1, 10)
    # labelsDummy = []
    # labelsDummy.append("time")
    # for coor in range(nCoordinatesAll):
    #     labelsDummy.append(coordinateSet.get(coor).getName())
    # numpy2storage(labelsDummy, dummyDataa, os.path.join(pathID,
    #                                                     "dummyData.sto"))


    # model_path = pathOpenSimModel
    # opensim.Logger.setLevel(4) # Logger level: error
    # model = opensim.Model(model_path)
    # model.initSystem()

    # # Get the force set and geometry set
    # forceSet = model.get_ForceSet()
    # geometrySet = model.get_ContactGeometrySet()

    # # Loop through the force set and select only the contact spheres
    # sphere_n = 0
    # center = np.zeros((6,3))
    # E = np.zeros(6)
    # c = np.zeros(6)
    # r = np.zeros(6)

    # for i in range(forceSet.getSize()):        
    #     force = forceSet.get(i)
    #     if (force.getConcreteClassName() == "SmoothSphereHalfSpaceForce"
    #         and force.getName()[-1] == 'r'
    #         ):
    #         force_obj = opensim.SmoothSphereHalfSpaceForce.safeDownCast(force)
        
    #         socket0Name = force.getSocketNames()[0]
    #         socket0 = force.getSocket(socket0Name)
    #         socket0_obj = socket0.getConnecteeAsObject()
    #         socket0_objName = socket0_obj.getName()            
    #         geo0 = geometrySet.get(socket0_objName)
    #         # geo0_loc = geo0.get_location().to_numpy()
    #         # geo0_or = geo0.get_orientation().to_numpy()
    #         # geo0_frameName = geo0.getFrame().getName()

    #         socket1Name = force.getSocketNames()[1]
    #         socket1 = force.getSocket(socket1Name)
    #         socket1_obj = socket1.getConnecteeAsObject()
    #         socket1_objName = socket1_obj.getName()            
    #         geo1 = geometrySet.get(socket1_objName)

    #         # Set location
    #         geo1_loc = geo1.get_location().to_numpy()
    #         # geo1_frameName = geo1.getFrame().getName()
    #         obj = opensim.ContactSphere.safeDownCast(geo1)

    #         # Set radius
    #         geo1_radius = obj.getRadius()

    #         # force_obj.set_viscous_friction(0)
    #         # force_obj.set_transition_velocity(0.002)

    #         # get sphere parameters and put them in some container
    #         center[sphere_n,:] = geo1_loc
    #         E[sphere_n] = force_obj.get_stiffness()
    #         c[sphere_n] = force_obj.get_dissipation()
    #         r[sphere_n] = geo1_radius
    #         static_friction = force_obj.get_static_friction()
    
    #         sphere_n = sphere_n + 1


    # # # print to check
    # # for sphere_name, sphere_parameters in sphere_data.items():
    # #     print( '\nSphere name:', sphere_name)
    # #     for key in sphere_parameters:
    # #         print(key+':', sphere_parameters[key])
    # # exit()

    # reporter = opensim.ForceReporter()
    # reporter.setName('reporter')
    # reporter.setModel(model)
    # reporter.includeConstraintForces(True)

    # calcn_r = model.getBodySet().get('calcn_r')
    # # calcn_l = model.getBodySet().get('calcn_l')
    # # toes_l = model.getBodySet().get('toes_l')
    # toes_r = model.getBodySet().get('toes_r')
    # ground = model.getGround()

    # # Pointkine analysis to get the COP
    # s1_r = opensim.PointKinematics()
    # s1_r.setDebugLevel(-1)
    # s1_r.setName('s1')
    # s1_r.setModel(model)
    # s1_r.setBody(calcn_r)
    # s1_r.setRelativeToBody(ground)
    # s1_r.setPoint(opensim.Vec3(center[0][0],center[0][1],center[0][2]))
    # s1_r.setPointName('r')

    # s2_r = opensim.PointKinematics()
    # s2_r.setDebugLevel(-1)
    # s2_r.setName('s2')
    # s2_r.setModel(model)
    # s2_r.setBody(calcn_r)
    # s2_r.setRelativeToBody(ground)
    # s2_r.setPoint(opensim.Vec3(center[1][0],center[1][1],center[1][2]))
    # s2_r.setPointName('r')

    # s3_r = opensim.PointKinematics()
    # s3_r.setDebugLevel(-1)
    # s3_r.setName('s3')
    # s3_r.setModel(model)
    # s3_r.setBody(calcn_r)
    # s3_r.setRelativeToBody(ground)
    # s3_r.setPoint(opensim.Vec3(center[2][0],center[2][1],center[2][2]))
    # s3_r.setPointName('r')

    # s4_r = opensim.PointKinematics()
    # s4_r.setDebugLevel(-1)
    # s4_r.setName('s4')
    # s4_r.setModel(model)
    # s4_r.setBody(calcn_r)
    # s4_r.setRelativeToBody(ground)
    # s4_r.setPoint(opensim.Vec3(center[3][0],center[3][1],center[3][2]))
    # s4_r.setPointName('r')

    # s5_r = opensim.PointKinematics()
    # s5_r.setDebugLevel(-1)
    # s5_r.setName('s5')
    # s5_r.setModel(model)
    # s5_r.setBody(toes_r)
    # s5_r.setRelativeToBody(ground)
    # s5_r.setPoint(opensim.Vec3(center[4][0],center[4][1],center[4][2]))
    # s5_r.setPointName('r')

    # s6_r = opensim.PointKinematics()
    # s6_r.setDebugLevel(-1)
    # s6_r.setName('s6')
    # s6_r.setModel(model)
    # s6_r.setBody(toes_r)
    # s6_r.setRelativeToBody(ground)
    # s6_r.setPoint(opensim.Vec3(center[5][0],center[5][1],center[5][2]))
    # s6_r.setPointName('r')



    # analysis = opensim.AnalyzeTool()
    # analysis.setModel(model)
    # analysis.setSolveForEquilibrium(False)
    # analysis.setLoadModelAndInput(True)
    # analysis.updAnalysisSet().adoptAndAppend(reporter)
    # analysis.updAnalysisSet().adoptAndAppend(s1_r)
    # analysis.updAnalysisSet().adoptAndAppend(s2_r)
    # analysis.updAnalysisSet().adoptAndAppend(s3_r)
    # analysis.updAnalysisSet().adoptAndAppend(s4_r)
    # analysis.updAnalysisSet().adoptAndAppend(s5_r)
    # analysis.updAnalysisSet().adoptAndAppend(s6_r)

    # analysis.addAnalysisSetToModel()
    # analysis.setResultsDir('forces')
    # analysis.setDebugLevel(-1)



    # path =   'InverseDynamics\dummyData.sto'
    # analysis.setName('dummy')
    # analysis.setCoordinatesFileName(path)

    # analysis.run()

    # # load sto file of forces
    # _, labels, data = readMotionFile(
    #     'forces/' + 'dummy' + '_reporter_forces.sto')


    # GRF_x_labels = ('SmoothSphereHalfSpaceForce_s1_r.Sphere.force.X',
    #                 'SmoothSphereHalfSpaceForce_s2_r.Sphere.force.X',
    #                 'SmoothSphereHalfSpaceForce_s3_r.Sphere.force.X',
    #                 'SmoothSphereHalfSpaceForce_s4_r.Sphere.force.X',
    #                 'SmoothSphereHalfSpaceForce_s5_r.Sphere.force.X',
    #                 'SmoothSphereHalfSpaceForce_s6_r.Sphere.force.X',
    #                 )
    # GRF_y_labels = ('SmoothSphereHalfSpaceForce_s1_r.Sphere.force.Y',
    #                 'SmoothSphereHalfSpaceForce_s2_r.Sphere.force.Y',
    #                 'SmoothSphereHalfSpaceForce_s3_r.Sphere.force.Y',
    #                 'SmoothSphereHalfSpaceForce_s4_r.Sphere.force.Y',
    #                 'SmoothSphereHalfSpaceForce_s5_r.Sphere.force.Y',
    #                 'SmoothSphereHalfSpaceForce_s6_r.Sphere.force.Y',
    #                 )
    # GRF_z_labels = ('SmoothSphereHalfSpaceForce_s1_r.Sphere.force.Z',
    #                 'SmoothSphereHalfSpaceForce_s2_r.Sphere.force.Z',
    #                 'SmoothSphereHalfSpaceForce_s3_r.Sphere.force.Z',
    #                 'SmoothSphereHalfSpaceForce_s4_r.Sphere.force.Z',
    #                 'SmoothSphereHalfSpaceForce_s5_r.Sphere.force.Z',
    #                 'SmoothSphereHalfSpaceForce_s6_r.Sphere.force.Z',
    #                 )


    # # Get indexes of desired forces
    # idx_time = labels.index('time')
    # idx_GRF_x = [labels.index(i) for i in GRF_x_labels]
    # idx_GRF_y = [labels.index(i) for i in GRF_y_labels]
    # idx_GRF_z = [labels.index(i) for i in GRF_z_labels]

    # data = np.array(data)

    # # taglia prima riga
    # # data = data[1:, :]

    # time = data[:, idx_time] # Seconds
    # GRF_x = data[:, idx_GRF_x] # Newtons
    # GRF_y = data[:, idx_GRF_y]
    # GRF_z = data[:, idx_GRF_z]

    # # total forces on each axis
    # GRF_x_all = np.sum(GRF_x, axis=1)
    # GRF_y_all = np.sum(GRF_y, axis=1)
    # GRF_z_all = np.sum(GRF_z, axis=1)

    # # forces on all axes for each sphere
    # GRF_s1 = np.array([GRF_x[:, 0], GRF_y[:, 0], GRF_z[:, 0]]).transpose()
    # GRF_s2 = np.array([GRF_x[:, 1], GRF_y[:, 1], GRF_z[:, 1]]).transpose()
    # GRF_s3 = np.array([GRF_x[:, 2], GRF_y[:, 2], GRF_z[:, 2]]).transpose()
    # GRF_s4 = np.array([GRF_x[:, 3], GRF_y[:, 3], GRF_z[:, 3]]).transpose()
    # GRF_s5 = np.array([GRF_x[:, 4], GRF_y[:, 4], GRF_z[:, 4]]).transpose()
    # GRF_s6 = np.array([GRF_x[:, 5], GRF_y[:, 5], GRF_z[:, 5]]).transpose()

    # # load the result of point analysis to get the COP
    # # Only right foot for now
    # ii = 'dummy'
    # _, _, s1 = readMotionFile('forces/' + ii + '_s1_r_pos.sto')
    # _, _, s2 = readMotionFile('forces/' + ii + '_s2_r_pos.sto')
    # _, _, s3 = readMotionFile('forces/' + ii + '_s3_r_pos.sto')
    # _, _, s4 = readMotionFile('forces/' + ii + '_s4_r_pos.sto')
    # _, _, s5 = readMotionFile('forces/' + ii + '_s5_r_pos.sto')
    # _, _, s6 = readMotionFile('forces/' + ii + '_s6_r_pos.sto')

    # s1 = np.array(s1) # meters
    # s2 = np.array(s2)
    # s3 = np.array(s3)
    # s4 = np.array(s4)
    # s5 = np.array(s5)
    # s6 = np.array(s6)

    # s1 = s1[:, 1:]
    # s2 = s2[:, 1:]
    # s3 = s3[:, 1:]
    # s4 = s4[:, 1:]
    # s5 = s5[:, 1:]
    # s6 = s6[:, 1:]
    # # metti la y a zero

    # s1[:, 1] = np.zeros(len(s1))
    # s2[:, 1] = np.zeros(len(s2))
    # s3[:, 1] = np.zeros(len(s3))
    # s4[:, 1] = np.zeros(len(s4))
    # s5[:, 1] = np.zeros(len(s5))
    # s6[:, 1] = np.zeros(len(s6))

    # # Calculate the COP
    # COP = np.zeros((len(GRF_y), 3))
    # M_hor = np.zeros((len(GRF_y), 3))
    # for k in range(len(GRF_y)): # iterate over time steps

    #     # if no vertical force, COP is null
    #     if sum(GRF_y[k, :]) < 1e-2:
    #         COP[k, :] = [0, 0, 0]
    #     else:
    #         COP[k, :] = (
    #             s1[k, :]*GRF_y[k, 0]
    #             + s2[k, :]*GRF_y[k, 1]
    #             + s3[k, :]*GRF_y[k, 2]
    #             + s4[k, :]*GRF_y[k, 3]
    #             + s5[k, :]*GRF_y[k, 4]
    #             + s6[k, :]*GRF_y[k, 5]
    #             )/(sum(GRF_y[k, :]))
    #     # calcolo momenti delle reazioni orizzontali
    #     M_hor[k, :] = (
    #         np.cross((s1[k, :] - COP[k, :]), GRF_s1[k, :]) 
    #         + np.cross((s2[k, :] - COP[k, :]), GRF_s2[k, :]) 
    #         + np.cross((s3[k, :] - COP[k, :]), GRF_s3[k, :])
    #         + np.cross((s4[k, :] - COP[k, :]), GRF_s4[k, :]) 
    #         + np.cross((s5[k, :] - COP[k, :]), GRF_s5[k, :])
    #         + np.cross((s6[k, :] - COP[k, :]), GRF_s6[k, :])
    #         )

    # F = ca.external('F', os.path.join(outputDir, 
    #     outputFilename + '.dll'))
    # path =   'InverseDynamics\dummyData.sto'
    # _, _, data = readMotionFile(path)
    # data = np.array(data)


    # k = 5 # random time instant
    # X = np.zeros((2*(data.shape[1]-1)))
    # X[::2] = data[0,1:]
    # static_friction = static_friction*np.ones(1)
    # vecInput = np.concatenate((X,E,c,r,static_friction))

    # output = (F(vecInput)).full().flatten()
    # # 
    # # print(vecInput)
    # # print(vecInput.shape)
    # # exit()

    # COPx = COP[k,0]
    # COPz = COP[k,2]
    # My = M_hor[k,1]

    # output_osim = np.array((GRF_x_all[k], GRF_y_all[k], GRF_z_all[k], COPx, My, COPz))

    # print('Output F:', output)
    # print('Output OpenSim:', output_osim)
    # diff_ID = np.abs(output_osim - output)
    # for i in range(diff_ID.shape[0]):
    #     if diff_ID[i] > 1e-6:            
    #         diff_ID_perc = diff_ID[i]/np.abs(output_osim[i])*100            
    #         if diff_ID_perc > 0.1:
    #             raise ValueError("GRF verification test failed")    
    # # assert(np.max(np.abs(ID_osim - ID_F)) < 1e-6), (
    # #     "Torque verification test failed")
    # print('GRF verification test passed')

# %% Generate c-code with external function (and its Jacobian).
def generateF(dim):
    import foo
    importlib.reload(foo)
    cg = ca.CodeGenerator('foo_jac')
    arg = ca.SX.sym('arg', dim)
    y,_,_ = foo.foo(arg)
    F = ca.Function('F',[arg],[y])
    cg.add(F)
    cg.add(F.jacobian())
    cg.generate()

# %% Build/compile external function.
def buildExternalFunction(filename, CPP_DIR, nInputs):       
    
    # %% Part 1: build expression graph (i.e., generate foo.py).
    pathMain = os.getcwd()
    pathBuildExpressionGraph = os.path.join(pathMain, 'buildExpressionGraph')
    pathBuild = os.path.join(pathMain, 'build-ExpressionGraph' + filename)
    os.makedirs(pathBuild, exist_ok=True)
    OpenSimAD_DIR = os.path.join(pathMain, 'opensimAD-install')
    os.makedirs(OpenSimAD_DIR, exist_ok=True)
    os_system = platform.system()
    
    if os_system == 'Windows':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'windows') 
        # OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'windows')      
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'modello_nostro')      

  
        BIN_DIR = os.path.join(OpenSimADOS_DIR, 'bin')
        SDK_DIR = os.path.join(OpenSimADOS_DIR, 'sdk')
        # Download libraries if not existing locally.
        if not os.path.exists(BIN_DIR):
            url = 'https://sourceforge.net/projects/opensimad/files/windows.zip'
            zipfilename = 'windows.zip'                
            download_file(url, zipfilename)
            with zipfile.ZipFile('windows.zip', 'r') as zip_ref:
                zip_ref.extractall(OpenSimAD_DIR)
            os.remove('windows.zip')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '"  -A x64 -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + SDK_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "cmake --build . --config RelWithDebInfo"
        
    elif os_system == 'Linux':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'linux')
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'linux')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'linux', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/linux.tar.gz'
            zipfilename = 'linux.tar.gz'                
            download_file(url, zipfilename)
            cmd_tar = 'tar -xf linux.tar.gz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('linux.tar.gz')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
        
    elif os_system == 'Darwin':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'macOS')
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'macOS')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'macOS', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/macOS.tgz'
            zipfilename = 'macOS.tgz'                
            download_file(url, zipfilename)
            cmd_tar = 'tar -xf macOS.tgz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('macOS.tgz')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
    
    os.chdir(pathBuild)    
    os.system(cmd1)    
    os.system(cmd2)
    
    if os_system == 'Windows':
        os.chdir(BIN_DIR)
        path_EXE = os.path.join(pathBuild, 'RelWithDebInfo', filename + '.exe')
        cmd2w = '"{}"'.format(path_EXE)
        os.system(cmd2w)
    
    # %% Part 2: build external function (i.e., build .dll).
    fooName = "foo.py"
    pathBuildExternalFunction = os.path.join(pathMain, 'buildExternalFunction')
    path_external_filename_foo = os.path.join(BIN_DIR, fooName)
    path_external_functions_filename_build = os.path.join(pathMain, 'build-ExternalFunction' + filename)
    path_external_functions_filename_install = os.path.join(pathMain, 'install-ExternalFunction' + filename)
    os.makedirs(path_external_functions_filename_build, exist_ok=True) 
    os.makedirs(path_external_functions_filename_install, exist_ok=True)
    shutil.copy2(path_external_filename_foo, pathBuildExternalFunction)
    
    sys.path.append(pathBuildExternalFunction)
    os.chdir(pathBuildExternalFunction)
    

    generateF(nInputs)
    
    if os_system == 'Windows':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -A x64 -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "cmake --build . --config RelWithDebInfo --target install"
    elif os_system == 'Linux':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "make install"
    elif os_system == 'Darwin':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "make install"
    
    os.chdir(path_external_functions_filename_build)
    os.system(cmd3)
    os.system(cmd4)    
    os.chdir(pathMain)
    
    if os_system == 'Windows':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'bin', filename + '.dll'), CPP_DIR)
    elif os_system == 'Linux':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.so'), CPP_DIR)
        os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.so'), os.path.join(CPP_DIR, filename + '.so'))
    elif os_system == 'Darwin':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.dylib'), CPP_DIR)
        os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.dylib'), os.path.join(CPP_DIR, filename + '.dylib'))
    
    os.remove(os.path.join(pathBuildExternalFunction, "foo_jac.c"))
    os.remove(os.path.join(pathBuildExternalFunction, fooName))
    os.remove(path_external_filename_foo)
    shutil.rmtree(pathBuild)
    shutil.rmtree(path_external_functions_filename_install)
    shutil.rmtree(path_external_functions_filename_build)    

# %% From storage file to numpy array.
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

# %% From storage file to DataFrame.
def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% From numpy array to storage file.
def numpy2storage(labels, data, storage_file):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    f.write('name %s\n' % storage_file)
    f.write('datacolumns %d\n' % data.shape[1])
    f.write('datarows %d\n' % data.shape[0])
    f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()
    
# %% Download file given url.
def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
