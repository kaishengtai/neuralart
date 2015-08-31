--
-- Convenience functions to replicate Caffe preprocessing
--

local means = { 104, 117, 123 }
local long_edge = 500

function preprocess(img)
    local w, h = img:size(3), img:size(2)
    if w < h then
        img = image.scale(img, long_edge * w / h, long_edge)
    else
        img = image.scale(img, long_edge, long_edge * h / w)
    end

    local copy = torch.Tensor(img:size())
    copy[1] = img[3]
    copy[2] = img[2]
    copy[3] = img[1]
    img = copy

    img:mul(255)
    for i = 1, 3 do
        img[i]:add(-means[i])
    end
    return img:view(1, 3, img:size(2), img:size(3))
end

function depreprocess(img)
    img = img:float():view(3, img:size(3), img:size(4))
    for i = 1, 3 do
        img[i]:add(means[i])
    end
    img:div(255)

    local copy = torch.FloatTensor(img:size())
    copy[1] = img[3]
    copy[2] = img[2]
    copy[3] = img[1]
    img = copy
    img:clamp(0, 1)
    return img
end
