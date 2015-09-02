--
-- An implementation of the method described in 'A Neural Algorithm of Artistic
-- Style' by Leon Gatys, Alexander Ecker, and Matthias Bethge.
--
-- http://arxiv.org/abs/1508.06576
--

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'paths'
require 'optim'
local pl = require('pl.import_into')()
local printf = pl.utils.printf

paths.dofile('inception.lua')
paths.dofile('images.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('A Neural Algorithm of Artistic Style')
cmd:text()
cmd:text('Options:')
cmd:option('--style',           'none',  'Path to style image')
cmd:option('--content',         'none',  'Path to content image')
cmd:option('--style_factor',     5e9,    'Trade-off factor between style and content')
cmd:option('--num_iters',        500,    'Number of iterations')
cmd:option('--size',             500,    'Length of image long edge (0 to use original content size)')
cmd:option('--display_interval', 20,     'Iterations between image displays (0 to suppress display)')
cmd:option('--smoothness',       6e-3,   'Total variation norm regularization strength (higher for smoother output)')
cmd:option('--init',            'image', '{image, random}. Initialization mode for optimized image.')
cmd:option('--backend',         'cunn',  '{cunn, cudnn}. Neural network CUDA backend.')
cmd:option('--optimizer',       'sgd',   '{sgd, lbfgs}. Optimization algorithm.')
local opt = cmd:parse(arg)
if opt.size <= 0 then
    opt.size = nil
end

local euclidean = nn.MSECriterion()
euclidean.sizeAverage = false
euclidean:cuda()

-- compute the Gramian matrix for input
function gram(input)
    local k = input:size(2)
    local flat = input:view(k, -1)
    local gram = torch.mm(flat, flat:t())
    return gram
end

function collect_activations(model, activation_layers, gram_layers)
    local activations, grams = {}, {}
    for i, module in ipairs(model.modules) do
        local name = module._name
        if name then
            if activation_layers[name] then
                local activation = module.output.new()
                activation:resize(module.output:nElement())
                activation:copy(module.output)
                activations[name] = activation
            end

            if gram_layers[name] then
                grams[name] = gram(module.output):view(-1)
            end
        end
    end
    return activations, grams
end

function style_grad(gen, orig_gram)
    local k = gen:size(2)
    local size = gen:nElement()
    local size_sq = size * size
    local gen_gram = gram(gen)
    local gen_gram_flat = gen_gram:view(-1)
    local loss = euclidean:forward(gen_gram_flat, orig_gram)
    local grad = euclidean:backward(gen_gram_flat, orig_gram)
                          :view(gen_gram:size())

    -- normalization helps improve the appearance of the generated image
    local norm = torch.abs(grad):mean() * size_sq
    if norm > 0 then
        loss = loss / norm
        grad:div(norm)
    end
    grad = torch.mm(grad, gen:view(k, -1)):view(gen:size())
    return loss, grad
end

function content_grad(gen, orig)
    local gen_flat = gen:view(-1)
    local loss = euclidean:forward(gen_flat, orig)
    local grad = euclidean:backward(gen_flat, orig):view(gen:size())
    local norm = torch.abs(grad):mean()
    if norm > 0 then
        loss = loss / norm
        grad:div(norm)
    end
    return loss, grad
end

-- total variation gradient
function total_var_grad(gen)
    local x_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {1, -2}, {2, -1}}]
    local y_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {2, -1}, {1, -2}}]
    local grad = gen.new():resize(gen:size()):zero()
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, x_diff)
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, y_diff)
    return grad
end

-- load model
local model = create_model('inception_caffe.th', opt.backend)
collectgarbage()

-- choose style and content layers
local style_weights = {
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

local content_weights = {
    ['inception_3a'] = 1,
    ['inception_4a'] = 1,
}

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
local img = preprocess(image.load(opt.content), opt.size):cuda()
model:forward(img)
local img_activations, _ = collect_activations(model, content_weights, {})

-- load style image
local art = preprocess(
    image.load(opt.style), math.max(img:size(3), img:size(4))
):cuda()
model:forward(art)
local _, art_grams = collect_activations(model, {}, style_weights)
art = nil
collectgarbage()

function opfunc(input)
    -- forward prop
    model:forward(input)

    -- backpropagate
    local loss = 0
    local grad = torch.CudaTensor(model.output:size()):zero()
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
    ):cuda()
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
        learningRate = 0.1,
        momentum = 0.9,
        dampening = 0.0,
    }
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
