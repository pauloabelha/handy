function [object,mesh] = readPCLToHectorMesh(filepath, translation, obj_idx, texture)
    if ~exist('texture','var')
        mesh.texture = rand(16,16,3);
    else
        mesh.texture = texture;
    end
    pcl = ReadPointCloud(filepath);    
    mesh.vertices = pcl.v;
    mesh.faces = pcl.f + 1;
    mesh.textureFaceCoordinates = rand(size(mesh.faces,1),6);    
    mesh.shadingType = 'flat';%or 'phong'
    mesh.diffuseReflectionConstant = 1;
    mesh.specularReflectionConstant = 1;
    mesh.shininessConstant = 3;
    mesh.ambientReflectionConstant = 0.1;
    
    object.objectLibraryIndex = obj_idx;
    object.translationVector = translation;
    objectRotation = [1,1,1,pi/4];
    object.rotationMatrix = axang2rotm(objectRotation);
end

