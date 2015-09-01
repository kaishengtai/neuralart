--
-- Inception architecture
-- Ref: Szegedy et al. 2015, Going Deeper with Convolutions
--

function nn.Module:name(name)
    self._name = name
    return self
end

function nn.Module:findByName(name)
    if self._name == name then return self end
    if self.modules ~= nil then
        for i = 1, #self.modules do
            local module = self.modules[i]:findByName(name)
            if module ~= nil then return module end
        end
    end
end

function nn.Sequential:subnetwork(name)
    local subnet = nn.Sequential()
    for i, module in ipairs(self.modules) do
        subnet:add(module)
        if module._name == name then
            break
        end
    end
    subnet:cuda()
    return subnet
end

function create_model(weights_file, backend)

    local nnlib
    local lrn  -- local response normalization module
    if backend == 'cudnn' then
        require 'cudnn'
        print('using cudnn backend')
        nnlib = cudnn
        lrn = cudnn.SpatialCrossMapLRN
    elseif backend == 'cunn' then
        require 'inn'
        print('using cunn backend')
        nnlib = nn
        lrn = inn.SpatialCrossResponseNormalization
    else
        error('unrecognized backend: ' .. backend)
    end

    local function inception(name, input_size, config)
        local concat = nn.Concat(2)
        local conv1 = nn.Sequential()
        conv1:add(nnlib.SpatialConvolution(input_size, config[1], 1, 1, 1, 1):name(name .. '/1x1'))
             :add(nnlib.ReLU(true))
        local conv3 = nn.Sequential()
        conv3:add(nnlib.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1):name(name .. '/3x3_reduce'))
             :add(nnlib.ReLU(true))
             :add(nnlib.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1):name(name .. '/3x3'))
             :add(nnlib.ReLU(true))
        local conv5 = nn.Sequential()
        conv5:add(nnlib.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1):name(name .. '/5x5_reduce'))
             :add(nnlib.ReLU(true))
             :add(nnlib.SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2):name(name .. '/5x5'))
             :add(nnlib.ReLU(true))
        local pool = nn.Sequential():name(name .. '/pool')
        if nnlib == cudnn then
            pool:add(nnlib.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
        else
            pool:add(nnlib.SpatialAveragePooling(3, 3, 1, 1))
        end
        pool:add(nnlib.SpatialConvolution(input_size, config[4], 1, 1, 1, 1):name(name .. '/pool_proj'))
            :add(nnlib.ReLU(true))
        concat:add(conv1):add(conv3):add(conv5):add(pool)
              :name(name)
        return concat
    end

    local model = nn.Sequential()
        :add(nnlib.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3):name('conv1/7x7_s2'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialAveragePooling(3, 3, 2, 2))
        :add(lrn(5, 0.0001, 0.75, 1.0))
        :add(nnlib.SpatialConvolution(64, 64, 1, 1, 1, 1):name('conv2/3x3_reduce'))
        :add(nnlib.ReLU(true))
        :add(nnlib.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1):name('conv2/3x3'))
        :add(nnlib.ReLU(true))
        :add(lrn(5, 0.0001, 0.75, 1.0))
        :add(nnlib.SpatialAveragePooling(3, 3, 2, 2))
        :add(inception('inception_3a', 192, {64, {96, 128}, {16, 32}, 32}))
        :add(inception('inception_3b', 256, {128, {128, 192}, {32, 96}, 64}))
        :add(nnlib.SpatialAveragePooling(3, 3, 2, 2))
        :add(inception('inception_4a', 480, {192, {96, 208}, {16, 48}, 64}))
        :add(inception('inception_4b', 512, {160, {112, 224}, {24, 64}, 64}))
        :add(inception('inception_4c', 512, {128, {128, 256}, {24, 64}, 64}))
        :add(inception('inception_4d', 512, {112, {144, 288}, {32, 64}, 64}))
        :add(inception('inception_4e', 528, {256, {160, 320}, {32, 128}, 128}))
        :add(nnlib.SpatialAveragePooling(3, 3, 2, 2))
        :add(inception('inception_5a', 832, {256, {160, 320}, {32, 128}, 128}))
        :add(inception('inception_5b', 832, {384, {192, 384}, {48, 128}, 128}))
        :add(nnlib.SpatialAveragePooling(7, 7, 1, 1))
        :add(nn.Dropout(0.4))
        :add(nn.View(1024):setNumInputDims(3))
        :add(nn.Linear(1024, 1000):name('loss3/classifier'))
        :add(nn.LogSoftMax())

    local weights = torch.load(weights_file)
    for i, module in ipairs(model:listModules()) do
        if module.weight then module.weight:copy(weights[i][1]) end
        if module.bias then module.bias:copy(weights[i][2]) end
    end

    local inception_modules = {
        'inception_3a', 'inception_3b',
        'inception_4a', 'inception_4b', 'inception_4c', 'inception_4d', 'inception_4e',
        'inception_5a', 'inception_5b',
    }

    if nnlib ~= cudnn then
        for i, name in ipairs(inception_modules) do
            model:findByName(name .. '/pool'):insert(nn.SpatialZeroPadding(1, 1, 1, 1), 1)
        end
    end

    model = model:subnetwork('inception_4e')
    collectgarbage()
    model:cuda()
    return model
end
