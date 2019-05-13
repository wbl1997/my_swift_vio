function distance= totalDistance(p_GB)
% p_GB=[1,2,3;4,5,6;7,8,9];
if(size(p_GB,1)==3)
    deltaXYZ= p_GB(:,2:end) - p_GB(:,1:end-1);
    distance = sum(sqrt(sum(deltaXYZ.^2, 1)))
else
    deltaXYZ= p_GB(2:end,:) - p_GB(1:end-1,:);
    distance = sum(sqrt(sum(deltaXYZ.^2, 2)))
end
end