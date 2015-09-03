--
-- VGG-19 model (http://arxiv.org/abs/1409.1556)
-- Max pooling layers have been replaced with average pooling
--

function create_vgg(weights_file, backend)

    local nnlib
    local lrn  -- local response normalization module
    if backend == 'cudnn' then
        require 'cudnn'
        print('using cudnn backend')
        nnlib = cudnn
    elseif backend == 'cunn' then
        print('using cunn backend')
        nnlib = nn
    else
        error('unrecognized backend: ' .. backend)
    end

    local model = nn.Sequential()
        :add(nnlib.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1):name('conv1_1'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1):name('conv1_2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialAveragePooling(2, 2, 2, 2))
        :add(nnlib.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1):name('conv2_1'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):name('conv2_2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialAveragePooling(2, 2, 2, 2))
        :add(nnlib.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1):name('conv3_1'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1):name('conv3_2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1):name('conv3_3'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1):name('conv3_4'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialAveragePooling(2, 2, 2, 2))
        :add(nnlib.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1):name('conv4_1'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv4_2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv4_3'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv4_4'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialAveragePooling(2, 2, 2, 2))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv5_1'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv5_2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv5_3'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1):name('conv5_4'))
        :add(nnlib.ReLU(true))

    local weights = torch.load(weights_file)
    for i, module in ipairs(model:listModules()) do
        if module.weight then module.weight:copy(weights[i][1]) end
        if module.bias then module.bias:copy(weights[i][2]) end
    end

    collectgarbage()
    model:cuda()
    return model
end
