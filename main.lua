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
cmd:option('-style',   'none', 'Path to style image')
cmd:option('-content', 'none', 'Path to content image')
cmd:option('-style_factor', 5e9, 'Trade-off factor between style and content')
local opt = cmd:parse(arg)

local euclidean = nn.MSECriterion()
euclidean.sizeAverage = false
euclidean:cuda()

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

-- load model
local model = create_model('inception_caffe.th')
collectgarbage()

local style_layers = {
    'conv1/7x7_s2',
    'conv2/3x3',
    'inception_3a',
    'inception_3b',
    'inception_4a',
    'inception_4b',
    'inception_4c',
    'inception_4d',
    'inception_4e',
}

local content_layers = {
    'inception_3a',
    'inception_4a',
}

local style_index, content_index = {}, {}
for i, name in ipairs(style_layers) do style_index[name] = true end
for i, name in ipairs(content_layers) do content_index[name] = true end


-- load reference images
local art = preprocess(image.load(opt.style)):cuda()
model:forward(art)
local _, art_grams = collect_activations(model, {}, style_index)

local img = preprocess(image.load(opt.content)):cuda()
model:forward(img)
local img_activations, _ = collect_activations(model, content_index, {})

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
        if name and content_index[name] then
            local c_loss, c_grad = content_grad(module.output, img_activations[name])
            --printf('[content]\t%s\t%.2e\n', name, c_loss)
            loss = loss + c_loss / #content_layers
            grad:add(1 / #content_layers, c_grad)
        end

        -- add style gradient
        if name and style_index[name] then
            local s_loss, s_grad = style_grad(module.output, art_grams[name])
            --printf('[style]\t%s\t%.2e\n', name, s_loss)
            loss = loss + opt.style_factor * s_loss / #style_layers
            grad:add(opt.style_factor / #style_layers, s_grad)
        end
        grad = module:backward(module_input, grad)
    end

    return loss, grad
end

local optim_state = {
    learningRate = 0.1,
    momentum = 0.9,
    dampening = 0.0,
}

local input = img

-- generate from noise
--local input = torch.randn(1, 3, img:size(3), img:size(4)):cuda()

-- optimize
local num_iters = 500
local timer = torch.Timer()
local output = depreprocess(input):double()
image.display(output)

local frames_dir = 'frames'
if not paths.dirp(frames_dir) then
    paths.mkdir(frames_dir)
end
image.save(paths.concat(frames_dir, '0.jpg'), output)
for i = 1, num_iters do
    local _, loss = optim.sgd(opfunc, input, optim_state)
    loss = loss[1]
    if i % 200 == 0 then
        optim_state.learningRate = 0.9 * optim_state.learningRate
    end

    if i % 10 == 0 then
        printf('iter %5d\tloss %8.2e\tlr %8.2e\ttime %4.1f\n',
            i, loss, optim_state.learningRate, timer:time().real)
    end

    if i <= 20 or i % 5 == 0 then
        output = depreprocess(input):double()
        if i % 50 == 0 then image.display(output) end
        image.save(paths.concat(frames_dir, i .. '.jpg'), output)
    end
end
output = depreprocess(input)
image.display(output)
