function [images,tags] = getImagesAndTags()
% senet50-ferplus-logits.mat comes from:
% http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/
load senet50-ferplus-logits.mat images wavLogits
[~,maxIdx] = cellfun(@(x) max(x(:)), wavLogits, 'Uni', 0);
[~,tags] = cellfun(@(x, y) ind2sub(size(x), y), wavLogits, maxIdx);
