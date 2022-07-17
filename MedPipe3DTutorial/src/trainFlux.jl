# we should have the files preprocessed already from Hdf5
using Pkg
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")

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

"""
First we need to iterate through data load it into Hdf5 and collect the sizes - sizes will be needed as neural network expect uniform sizes of all training and test cases so we will need to get biggest image and pad others accordingly
code is based on https://github.com/Dale-Black/MedicalTutorials.jl/tree/master/src/3D_Segmentation/Heart


"""


#pathToHDF55="/media/jakub/NewVolume/projects/bigDataSet.hdf5"
pathToHDF5="/home/sliceruser/data/bigDataSet.hdf5"

fid = h5open(pathToHDF55, "r+")
max_epochs=3
val_interval=2
trainingKeys= keys(fid)[1:5]
valKeys= keys(fid)[6:8]


function loadfn_image(groupKey)
    gr= getGroupOrCreate(fid, string(groupKey))  
    return gr["image"][:,:,:,1,1]
end


function loadfn_label(groupKey)
    gr= getGroupOrCreate(fid, string(groupKey))  
    return  gr["labelSet"][:,:,:,1,1]
end




data_image(trainingKeys) = MLUtils.mapobs(loadfn_image,trainingKeys)
data_label(trainingKeys) =  MLUtils.mapobs(loadfn_label, trainingKeys)
data = (
    data_image(trainingKeys),
    data_label(trainingKeys),
)


#gpu_train_loader = Flux.DataLoader(data, batchsize = 16)

# testmethod =   BlockTask(
#     (Image{3}(), Label(1:2)),
#     (
#         ProjectiveTransforms((336, 336,352)),
#         ImagePreprocessing(),
#         OneHot()
#     )
# )

# image, mask = sample = MLUtils.getobs(data, 1);
# blocks=(FastAI.Image{3}(), FastAI.Label{UInt32}([UInt32(0),UInt32(1)]))
# task = ImageClassificationSingle(blocks)


function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end



conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())

conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))


function unet3D(in_chs, lbl_chs)
    # Contracting layers
    l1 = Chain(conv1(in_chs, 4))
    l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
    l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
    l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
    l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

    # Expanding layers
    l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
    l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))
    l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))
    l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))
    l10 = Chain(l9, conv1(4, lbl_chs))
end

model = unet3D(1, 2)|> gpu
optimizer = Flux.ADAM(0.01)
ps = Flux.params(model);
loss_function = Flux.Losses.dice_coeff_loss



maxEpoch=2
for epoch in 1:maxEpoch
    print("epoch ",epoch)
    @showprogress for groupKey in trainingKeys
        print("groupKey ",groupKey)
        gr= getGroupOrCreate(fid, groupKey)    
        x= gr["image"][:,:,:,:,:]
        y= Flux.onehotbatch(gr["labelSet"][:,:,:,:,:],0:1 )
        print("hd loaded")
        x, y = x |> gpu, y |> gpu
        print("passed on gpu")
        gs = Flux.gradient(ps) do
                ŷ = model(x)
                l=loss(ŷ, y)
                print(l)
                l
            end

        Flux.Optimise.update!(opt, ps, gs)
    end
end


modelpath = joinpath("/home/sliceruser/data", "model.bson")
let model = cpu(model) ## return model to cpu before serialization
    BSON.@save modelpath model 1
end
    
gr= getGroupOrCreate(fid, "1")
arr=gr["image"][:,:,:,:,:]
arr|>gpu


gpu_train_loader = Flux.DataLoader(mapobs(gpu, (xtrain, ytrain)), batchsize = 16)


train!(loss_function, ps, data, optimizer)

# dls = data
# #model = taskmodel(task, Models.xresnet18())
# lossfn = tasklossfn(task)
# learner = Learner(model, dls, ADAM(), loss_function, Metrics(accuracy))#ToGPU()




train_loader = Flux.DataLoader((data_image(trainingKeys), data_label(trainingKeys)), batchsize = 1, shuffle = true)
# ... model, optimizer and loss definitions
for epoch in 1:nepochs
    for (xtrain_batch, ytrain_batch) in train_loader
        print(MLUtils.getobs(xtrain_batch, 1))

        # x, y = gpu(xtrain_batch), gpu(ytrain_batch)
        # gradients = gradient(() -> loss(x, y), parameters)
        # Flux.Optimise.update!(optimizer, parameters, gradients)
    end
end