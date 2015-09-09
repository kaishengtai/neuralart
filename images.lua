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
    end

    local w, h = img:size(3), img:size(2)
    if scale then
        if w < h then
            img = image.scale(img, scale * w / h, scale)
        else
            img = image.scale(img, scale, scale * h / w)
        end
    end

    -- reverse channels
    local copy = torch.FloatTensor(img:size())
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
    local copy = torch.FloatTensor(3, img:size(3), img:size(4)):copy(img)
    for i = 1, 3 do
        copy[i]:add(means[i])
    end
    copy:div(255)

    -- reverse channels
    local copy2 = torch.FloatTensor(copy:size())
    copy2[1] = copy[3]
    copy2[2] = copy[2]
    copy2[3] = copy[1]
    copy2:clamp(0, 1)
    return copy2
end
