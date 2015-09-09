--
-- An implementation of the method described in 'A Neural Algorithm of Artistic
-- Style' by Leon Gatys, Alexander Ecker, and Matthias Bethge.
--
-- http://arxiv.org/abs/1508.06576
--

require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'optim'
local pl = require('pl.import_into')()
local printf = pl.utils.printf

local cmd = torch.CmdLine()
cmd:text()
cmd:text('A Neural Algorithm of Artistic Style')
cmd:text()
cmd:text('Options:')
cmd:option('--model',           'vgg',    '{inception, vgg}. Model to use.')
cmd:option('--style',           'none',   'Path to style image')
cmd:option('--content',         'none',   'Path to content image')
cmd:option('--style_factor',     2e9,     'Trade-off factor between style and content')
cmd:option('--num_iters',        500,     'Number of iterations')
cmd:option('--size',             500,     'Length of image long edge (0 to use original content size)')
cmd:option('--display_interval', 20,      'Iterations between image displays (0 to suppress display)')
cmd:option('--smoothness',       0,       'Total variation norm regularization strength (higher for smoother output)')
cmd:option('--init',            'image',  '{image, random}. Initialization mode for optimized image.')
cmd:option('--backend',         'cunn',   '{cunn, cudnn}. Neural network CUDA backend.')
cmd:option('--optimizer',       'lbfgs',  '{sgd, lbfgs}. Optimization algorithm.')
cmd:option('--cpu',              false,   'Optimize on CPU (only with VGG network).')
opt = cmd:parse(arg)
if opt.size <= 0 then
    opt.size = nil
end

if not opt.cpu then
    require 'cutorch'
    require 'cunn'
end

paths.dofile('models/util.lua')
paths.dofile('models/vgg19.lua')
paths.dofile('models/inception.lua')
paths.dofile('images.lua')
paths.dofile('costs.lua')

-- check for model files
local inception_path = 'models/inception_caffe.th'
local vgg_path = 'models/vgg_normalized.th'
if opt.model == 'inception' then
    if not paths.filep(inception_path) then
        print('ERROR: could not find Inception model weights at ' .. inception_path)
        print('run download_models.sh to download model weights')
        error('')
    end

    if opt.cpu then
        error('CPU optimization only works with VGG model')
    end
elseif opt.model == 'vgg' then
    if not paths.filep(vgg_path) then
        print('ERROR: could not find VGG model weights at ' .. vgg_path)
        print('run download_models.sh to download model weights')
        error('')
    end
else
    error('invalid model: ' .. opt.model)
end

-- load model
local model, style_weights, content_weights
if opt.model == 'inception' then
    style_weights = {
        ['conv1/7x7_s2'] = 1,
        ['conv2/3x3']    = 1,
        ['inception_3a'] = 1,
        ['inception_3b'] = 1,
        ['inception_4a'] = 1,
        ['inception_4b'] = 1,
        ['inception_4c'] = 1,
        ['inception_4d'] = 1,
        ['inception_4e'] = 1,
    }

    content_weights = {
        ['inception_3a'] = 1,
        ['inception_4a'] = 1,
    }

    model = create_inception(inception_path, opt.backend)
elseif opt.model == 'vgg' then
    style_weights = {
        ['conv1_1'] = 1,
        ['conv2_1'] = 1,
        ['conv3_1'] = 1,
        ['conv4_1'] = 1,
        ['conv5_1'] = 1,
    }

    content_weights = {
        ['conv4_2'] = 1,
    }

    model = create_vgg(vgg_path, opt.backend)
end

-- run on GPU
if opt.cpu then
    model:float()
else
    model:cuda()
end
collectgarbage()

-- compute normalization factor
local style_weight_sum = 0
local content_weight_sum = 0
for k, v in pairs(style_weights) do
    style_weight_sum = style_weight_sum + v
end

for k, v in pairs(content_weights) do
    content_weight_sum = content_weight_sum + v
end

-- load content image
local img = preprocess(image.load(opt.content), opt.size)
if not opt.cpu then
    img = img:cuda()
end
model:forward(img)
local img_activations, _ = collect_activations(model, content_weights, {})

-- load style image
local art = preprocess(
    image.load(opt.style), math.max(img:size(3), img:size(4))
)
if not opt.cpu then
    art = art:cuda()
end
model:forward(art)
local _, art_grams = collect_activations(model, {}, style_weights)
art = nil
collectgarbage()

function opfunc(input)
    -- forward prop
    model:forward(input)

    -- backpropagate
    local loss = 0
    local grad = opt.cpu and torch.FloatTensor() or torch.CudaTensor()
    grad:resize(model.output:size()):zero()
    for i = #model.modules, 1, -1 do
        local module_input = (i == 1) and input or model.modules[i - 1].output
        local module = model.modules[i]
        local name = module._name

        -- add content gradient
        if name and content_weights[name] then
            local c_loss, c_grad = content_grad(module.output, img_activations[name])
            local w = content_weights[name] / content_weight_sum
            --printf('[content]\t%s\t%.2e\n', name, w * c_loss)
            loss = loss + w * c_loss
            grad:add(w, c_grad)
        end

        -- add style gradient
        if name and style_weights[name] then
            local s_loss, s_grad = style_grad(module.output, art_grams[name])
            local w = opt.style_factor * style_weights[name] / style_weight_sum
            --printf('[style]\t%s\t%.2e\n', name, w * s_loss)
            loss = loss + w * s_loss
            grad:add(w, s_grad)
        end
        grad = module:backward(module_input, grad)
    end

    -- total variation regularization for denoising
    grad:add(total_var_grad(input):mul(opt.smoothness))
    return loss, grad:view(-1)
end

-- image to optimize
local input
if opt.init == 'image' then
    input = img
elseif opt.init == 'random' then
    input = preprocess(
        torch.randn(3, img:size(3), img:size(4)):mul(0.1):add(0.5):clamp(0, 1)
    )

    if not opt.cpu then
        input = input:cuda()
    end
else
    error('unrecognized initialization option: ' .. opt.init)
end

local timer = torch.Timer()
local output = depreprocess(input):double()
if opt.display_interval > 0 then
    image.display(output)
end

-- make directory to save intermediate frames
local frames_dir = 'frames'
if not paths.dirp(frames_dir) then
    paths.mkdir(frames_dir)
end
image.save(paths.concat(frames_dir, '0.jpg'), output)

-- set optimizer options
local optim_state
if opt.optimizer == 'sgd' then
    optim_state = {
        momentum = 0.9,
        dampening = 0.0,
    }

    if opt.model == 'inception' then
        optim_state.learningRate = 5e-2
    else
        optim_state.learningRate = 1e-3
    end
elseif opt.optimizer == 'lbfgs' then
    optim_state = {
        maxIter = 3,
        learningRate = 1,
    }
else
    error('unknown optimizer: ' .. opt.optimizer)
end

-- optimize
for i = 1, opt.num_iters do
    local _, loss = optim[opt.optimizer](opfunc, input, optim_state)
    loss = loss[1]

    -- anneal learning rate
    if opt.optimizer == 'sgd' and i % 100 == 0 then
        optim_state.learningRate = 0.75 * optim_state.learningRate
    end

    if i % 10 == 0 then
        printf('iter %5d\tloss %8.2e\tlr %8.2e\ttime %4.1f\n',
            i, loss, optim_state.learningRate, timer:time().real)
    end

    if i <= 20 or i % 5 == 0 then
        output = depreprocess(input):double()
        if opt.display_interval > 0 and i % opt.display_interval == 0 then
            image.display(output)
        end
        image.save(paths.concat(frames_dir, i .. '.jpg'), output)
    end
end

output = depreprocess(input)
if opt.display_interval > 0 then
    image.display(output)
end
