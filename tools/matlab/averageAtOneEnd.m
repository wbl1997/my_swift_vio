function [average, startIndex, finishIndex] = averageAtOneEnd(...
    matrix, windowSize, columnIndices, tail, timeTol)
% take the average of data at the head or tail within windowSize.
% data has timestamps in sec in the first column.
% windowSize is the duration to take average.
% columnIndices columns to take average.
% tail: 0 head 1 tail.
% timeTol: tolerance to associate two timestamps.

if nargin < 5
    timeTol = 5e-2;
end
if nargin < 4
    tail = 1;
end
startIndex = 1;
finishIndex = size(matrix, 1);
if tail == 1
    startIndex = find(abs(matrix(end, 1) - windowSize - matrix(:,1)) <...
        timeTol, 1);
    if isempty(startIndex)
        startIndex = max(1, size(matrix, 1) - 100);
    end
else
    finishIndex = find(abs(matrix(1, 1) + windowSize - matrix(:,1)) <...
        timeTol, 1);
    if isempty(finishIndex)
        finishIndex = min(size(matrix, 1), 100);
    end
end
average = mean(matrix(startIndex:finishIndex, columnIndices), 1);
end
