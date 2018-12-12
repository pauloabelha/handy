function [mesh] = readYCBObjMesh(filepath)
    texture = imread([filepath 'texture_map.png']);
    obj = readObj([filepath 'textured.obj']);    
    
    mesh.vertices = obj.v;
    mesh.faces = obj.f.v;
    mesh.textureVertexCoordinates = obj.vt;
    mesh.texture = texture;
    mesh.shadingType = 'flat';%or 'phong'
    mesh.diffuseReflectionConstant = 1;
    mesh.specularReflectionConstant = 1;
    mesh.shininessConstant = 3;
    mesh.ambientReflectionConstant = 0.1;   
    
end

