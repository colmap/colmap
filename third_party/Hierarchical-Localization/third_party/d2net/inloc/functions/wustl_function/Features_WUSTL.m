function [f, d] = features_custom(I_path)
  data = load([I_path '.d2-net'], '-mat');
  f = double(data.keypoints(:, 1 : 3).');
  d = double(data.descriptors.');
end

