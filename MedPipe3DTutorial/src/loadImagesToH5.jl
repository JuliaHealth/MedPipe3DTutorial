using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5,Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using UNet
using Hyperopt,Plots
using MedPipe3D.LoadFromMonai
using Flux
using Distributed
#]add Flux Hyperopt Plots UNet MedEye3d Distributions Clustering IrrationalConstants ParallelStencil CUDA HDF5 MedEval3D MedPipe3D Colors
CUDA.allowscalar(true)



#downloaded from https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

#directory where we want to store our HDF5 that we will use
pathToHDF5="/home/sliceruser/data/bigDataSet.hdf5"
#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
fid = h5open(pathToHDF5, "w")
root_dir = "/home/sliceruser/data/Spleen/Task09_Spleen"
targetSpacing=(1.5,1.5,1.5)

# resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
# compressed_file = joinpath(root_dir, "Task09_Spleen.tar")
# md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
# monai=MedPipe3D.LoadFromMonai.getMonaiObject()
# monai.apps.download_and_extract(resource, compressed_file, root_dir, md5)


train_labels = map(fileEntry-> joinpath(root_dir,"labelsTr",fileEntry),readdir(joinpath(root_dir,"labelsTr"); sort=true))
train_images = map(fileEntry-> joinpath(root_dir,"imagesTr",fileEntry),readdir(joinpath(root_dir,"imagesTr"); sort=true))

#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
# the paths are for some reasons reading incorrectly
zipped=map(tupl -> (replace(tupl[1], "._" => ""), replace(tupl[2], "._" => "")),zipped)
tupl=zipped[1]

#loaded = LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2],targetSpacing)

#proper loading using some utility function

sizes=Vector{Tuple{Int64, Int64, Int64}}(undef,length(zipped))
noError=falses(length(zipped))#Vector{Bool}(false,length(zipped))
chosen = zipped#[ifLiverPresent]

for (indexx,tupl) in enumerate(zipped)
        print("index $indexx ")
        #try
            loaded = LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2],targetSpacing )
            noError[indexx]=true
            sizes[indexx]=size(loaded[1])
            print(size(loaded[1]))
        #catch
         #   print("error with $indexx")
        #end    
end    




sum(noError)
# we get the maximum size so we will get in the en uniform size
chosen = zipped[noError]

maxX=  Int(ceil(maximum(map(tupl->tupl[1],sizes))/16)*16)
maxY=  Int(ceil(maximum(map(tupl->tupl[2],sizes))/16)*16)
maxZ=  Int(ceil(maximum(map(tupl->tupl[3],sizes))/16)*16)



print("maxes ($maxX , $maxY , $maxZ  )") #(336 , 336 , 352  )
# print(chosen)



#we save transformed data into hdf5 so we will not need to transform it on loading data into flux
#one could consider doing it in parallel but would need to setup parallel HDF5 https://juliaio.github.io/HDF5.jl/stable/#Parallel-HDF5
for (indexx,tupl) in enumerate(chosen)
    print("index loading  $indexx ")
        patienGroupName=string(indexx)
        #if the image was already loaded earlier we can ignore it
        if(!haskey(fid, patienGroupName))
            loaded = LoadFromMonai.loadandPad(tupl[1],tupl[2],targetSpacing, (maxX,maxY, maxZ) )
            gr= getGroupOrCreate(fid, patienGroupName)
            #we are intrested only about liver
            labelArr=loaded[2]
            #there are some artifacts on edges
            labelArr[1,:,:].=0
            labelArr[:,1,:].=0
            labelArr[:,:,1].=0

            labelArr[maxX,:,:].=0
            labelArr[:,maxY,:].=0
            labelArr[:,:,maxZ].=0
            #typeof(loaded[1])

            imageArr= reshape( loaded[1],(maxX,maxY, maxZ, 1, 1))
            labelArr= reshape( labelArr,(maxX,maxY, maxZ, 1, 1))

            saveMaskBeforeVisualization(fid,patienGroupName,imageArr,"image", "CT" )
            saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )
            writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])
    end
end 

keys(fid)

close(fid)


#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
fid = h5open(pathToHDF5, "r+")

keys(fid)

close(fid)
