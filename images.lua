--
-- Convenience functions to replicate Caffe preprocessing
--

local means = { 104, 117, 123 }

function preprocess(img, scale)
    -- handle monochrome images
    if img:size(1) == 1 then
        local copy = torch.FloatTensor(3, img:size(2), img:size(3))
        copy[1] = img[1]
        copy[2] = img[1]
        copy[3] = img[1]
        img = copy
    elseif img:size(1) == 4 then
        img = img[{{1,3},{},{}}]
    end

    local w, h = img:size(3), img:size(2)
    if scale then
        if w < h then
            img = image.scale(img, scale * w / h, scale)
        else
            img = image.scale(img, scale, scale * h / w)
        end
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
