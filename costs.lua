--
-- Cost functions
--

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

--
-- gradient computation functions
--

local euclidean = nn.MSECriterion()
euclidean.sizeAverage = false
euclidean:cuda()

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
    local norm = size_sq
    if opt.model == 'inception' then
        norm = torch.abs(grad):mean() * size_sq
    else
        norm = size_sq
    end
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
    if opt.model == 'inception' then
        local norm = torch.abs(grad):mean()
        if norm > 0 then
            loss = loss / norm
            grad:div(norm)
        end
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
