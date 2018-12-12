function [YCBMesh] = saveYCBObjAsMatFile(filepath, saveName)
    YCBMesh = readYCBObjMesh(filepath);
    save(saveName);
end

