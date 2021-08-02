function [d, foot] = pointToLine(pt, v1, v2)
% pt should be nx3
% v1 and v2 are vertices on the line (each 1x3)
% d is a nx1 vector with the orthogonal distances
% foot are the perpendicular foots of pt on the line, nx3 
v1 = repmat(v1,size(pt,1),1);
v2 = repmat(v2,size(pt,1),1);
a = v1 - v2;
b = pt - v2;
d = sqrt(sum(cross(a,b,2).^2,2)) ./ sqrt(sum(a.^2,2));

direction = v2(1, :) - v1(1, :);
coeff_direction = direction / (direction * direction');
lambda = (pt - v1) * coeff_direction';
foot = v1 + lambda * direction;