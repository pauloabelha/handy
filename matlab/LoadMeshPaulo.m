function [ mesh ] = LoadMeshPaulo( fileName, texturefilepath, scale )
%LOADMESH Summary of this function goes here
%   Detailed explanation goes here
disp(['Reading PLY...' fileName]);
[TRI,PTS, DATA, COMMENTS] = ply_read(fileName,'tri');

if exist('scale','var') && strcmp(scale,'meters')
    PTS = PTS * 1000;
    DATA.vertex.x = DATA.vertex.x * 1000;
    DATA.vertex.y = DATA.vertex.y * 1000;
    DATA.vertex.z = DATA.vertex.z * 1000;
end

mesh.faces = TRI';
mesh.vertices = PTS';
if isfield(DATA.vertex,'texture_u')
    mesh.textureVertexCoordinates = [DATA.vertex.texture_u,DATA.vertex.texture_v];
end

if isfield(DATA.face,'texcoord')
    mesh.textureFaceCoordinates = cell2mat(DATA.face.texcoord);
end

disp(['Reading texture...' texturefilepath]);
mesh.texture = imread(texturefilepath);


end

