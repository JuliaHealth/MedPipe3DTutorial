# we should have the files preprocessed already from Hdf5
using Pkg
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
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
using Hyperopt,Plots
using MedPipe3D.LoadFromMonai
using Flux,MLUtils
using MLDataPattern
using ImageCore
using DataLoaders
# using CUDA
using FastAI,DataAugmentation
using StaticArrays
using ProgressMeter: @showprogress
using Flux,FastAI,DataAugmentation,DLPipelines,MLDataPattern,ImageCore
using DataAugmentation: OneHot, Image
using MLUtils
CUDA.allowscalar(true)
using CUDA: CuIterator
using BSON
include("Unet.jl")
using Pkg
using Random
using Unzip
#Pkg.add(url="https://github.com/DhairyaLGandhi/UNet.jl.git")
#using UNet
using Base.Iterators
"""
First we need to iterate through data load it into Hdf5 and collect the sizes - sizes will be needed as neural network expect uniform sizes of all training and test cases so we will need to get biggest image and pad others accordingly
code is based on https://github.com/Dale-Black/MedicalTutorials.jl/tree/master/src/3D_Segmentation/Heart
"""

pathToHDF55="/media/jakub/NewVolume/projects/bigDataSet.hdf5"
modelpath = joinpath("/media/jakub/NewVolume/projects", "modelB.bson")

#pathToHDF5="/home/sliceruser/data/bigDataSet.hdf5"

fid = h5open(pathToHDF55, "r+")
max_epochs=3
val_interval=2
trainingKeys= keys(fid)[1:80]
valKeys= keys(fid)[6:8]
batchSize=16

model = UNet( 1,1,stages=4) |> gpu

optimizer = Flux.ADAM(0.01)
parameters = Flux.params(model);
loss_function = Flux.Losses.dice_coeff_loss

"""
Given 3D data divides it into batches of 2D slices across Z dimension
it ignores last not full batch
"""
function divideToBatchInZ(fid,groupKey)
    gr= getGroupOrCreate(fid, groupKey) 
    imageArr=gr["image"][:,:,:]
    labelArr=gr["labelSet"][:,:,:]
    return filter(tupl->sum(tupl[2])>0 ,getSlices(imageArr,labelArr) )
end #divideToBatchInZ

"""
get slices out of image and reshape them
"""
function getSlices(imageArr,labelArr)
    sizz= size(imageArr)
    return map(index-> (reshape(imageArr[:,:,index], (size(imageArr[:,:,index])...,1,1)) 
                        ,reshape(labelArr[:,:,index], (size(labelArr[:,:,index])...,1,1)))
                , 1:sizz[3]   )
    end

"""
given list of imagesconcatenates slices that contain any posiive entry

"""
function fuseSlicesMultiImage(imageNamesList,fid )
    list= map(groupKey-> divideToBatchInZ(fid,groupKey),imageNamesList)
    return cat(list...,dims=1)
end    


"""
take list of slices and create batches from it
"""
function divideIntoBatches(listSlices, batchSize)
    sizz=size(listSlices)
    floored= Int(floor(sizz[1]/batchSize))-1
    return map(i->batchIm=cat(listSlices[(i-1)*batchSize:(i-1)*batchSize]...,dims=4),1:floored ) 
end

#get all slices with some spleen on it
posLayers=fuseSlicesMultiImage(trainingKeys,fid )
nepochs=650

# posLayers=unzip(posLayers)
#train_loader = Flux.DataLoader((posLayers[1], posLayers[2]), batchsize = 8, shuffle = true)
for epoch in 1:nepochs    
    posLayers=shuffle(posLayers)
    for (batch, (x,y)) in enumerate(CuIterator(posLayers))
    # for (xtrain_batch, ytrain_batch) in train_loader
    #     x, y = gpu(xtrain_batch), gpu(ytrain_batch)
        gradients = gradient(() -> loss_function(model(x), y), parameters)
        Flux.Optimise.update!(optimizer, parameters, gradients)
    end
    print(" epoch ",epoch)
end

using BSON
let model = cpu(model) ## return model to cpu before serialization
    BSON.@save modelpath model
end





using BSON: @load
@load modelpath model


using MedPipe3D
exampleKey= keys(fid)[81]
# divided = divideToBatchInZ(batchSize,fid,exampleKey)
gr= getGroupOrCreate(fid, exampleKey) 
imageArr=gr["image"][:,:,:]
labelArr=gr["labelSet"][:,:,:]

slicessIm=getSlices(imageArr,labelArr)
slicessIm=unzip(slicessIm)
slicessIm=slicessIm[1]

modelOutput= map(batch-> model(batch)   ,slicessIm) 
modelOutput=modelOutput |>cpu 
modelOutputCat=cat(map(el->el[:,:,1,:], modelOutput)..., dims=3)
sizz_out=size(modelOutputCat)
grr= getGroupOrCreate(fid, exampleKey) 
imageArr=Int32.(round.(grr["image"][:,:,1:sizz_out[3]]))
labelArr=grr["labelSet"][:,:,1:sizz_out[3]]


toSaveKey="506"
grr= getGroupOrCreate(fid, toSaveKey) 
writeGroupAttribute(fid,toSaveKey, "spacing", [1.5,1.5,1.5])

saveMaskBeforeVisualization(fid,toSaveKey,imageArr,"image", "CT" )
saveMaskBeforeVisualization(fid,toSaveKey,labelArr,"labelSet", "boolLabel" )

listOfColorUsed= falses(18)

#manual Modification array
algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,1)
    addTextSpecs[1]=algoVisualization

mainScrollDat= loadFromHdf5Prim(fid,toSaveKey,addTextSpecs,listOfColorUsed)

algoOutput= getArrByName("algoOutput" ,mainScrollDat)
algoOutput[:,:,:]=modelOutputCat

saveMaskbyName(fid,toSaveKey , mainScrollDat, "algoOutput", "contLabel")

close(fid)




# using MedEye3d

# pathToHDF55="/media/jakub/NewVolume/projects/bigDataSet.hdf5"
# modelpath = joinpath("/media/jakub/NewVolume/projects", "modelB.bson")
# listOfColorUsed= falses(18)

# toSaveKey="506"
# fid = h5open(pathToHDF55, "r+")


# #manual Modification array
# algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
#     name = "algoOutput",
#     # we point out that we will supply multiple colors
#     isContinuusMask=true,
#     colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
#     ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
#    )

#     addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,1)
#     addTextSpecs[1]=algoVisualization

# mainScrollDat= loadFromHdf5Prim(fid,toSaveKey,addTextSpecs,listOfColorUsed)
