function [object, mesh] = generateCube(side_size, center, obj_idx)
    if ~exist('center','var')
       center = [0, 0, 0]; 
    end
    mesh.vertices = [side_size,side_size,side_size;...
                     side_size,side_size,-side_size;...
                     side_size,-side_size,side_size;...
                     side_size,-side_size,-side_size;...
                     -side_size,side_size,side_size;...
                     -side_size,side_size,-side_size;...
                     -side_size,-side_size,side_size;...
                     -side_size,-side_size,-side_size];
    mesh.vertices = 0.1*mesh.vertices/2;
    mesh.faces = convhulln(mesh.vertices);
    faceNormals = cross(mesh.vertices(mesh.faces(:,2),:)-mesh.vertices(mesh.faces(:,1),:),...
        mesh.vertices(mesh.faces(:,3),:)-mesh.vertices(mesh.faces(:,2),:),2);
    faceMeans = (mesh.vertices(mesh.faces(:,1),:)+mesh.vertices(mesh.faces(:,2),:)+mesh.vertices(mesh.faces(:,3),:))/3;
    faceOrientations = sign(dot(faceNormals,faceMeans,2));
    for ii = 1:size(mesh.faces,1)
        if faceOrientations(ii) < 0 
            mesh.faces(ii,:) = mesh.faces(ii,[1,3,2]);
        end
    end
    % Random texture
    mesh.textureFaceCoordinates = rand(size(mesh.faces,1),6);
    mesh.texture = rand(16,16,3);
    mesh.shadingType = 'flat';%or 'phong'
    mesh.diffuseReflectionConstant = 1;
    mesh.specularReflectionConstant = 1;
    mesh.shininessConstant = 3;
    mesh.ambientReflectionConstant = 0.1;
    %% Define object pose
    object.objectLibraryIndex = obj_idx;
    object.translationVector = center;
    objectRotation = [1,1,1,pi/4];
    object.rotationMatrix = axang2rotm(objectRotation);
end

