function errors=getRMSEAt(times)
% show rmse for specific epochs since the beginning
if(nargin==0)
    times=[0,15,45,135];
end
path = 'G:\temp';
rmse = load([path, '\sinusoidRMSE.txt']);
if(~isempty(rmse))
  rmse(:,1) = rmse(:,1) - rmse(1,1);
end
assert(size(rmse,2)==56);
errors = zeros(length(times), 14);
jack=1;
for i=1: length(times)
    epoch= times(i);
    while(rmse(jack,1)<epoch)
        jack=jack+1;
    end
    assert(rmse(jack,1)-epoch<1e-8);
    error_bg = rmse(jack, 11:13)*rmse(jack, 11:13)';
    error_ba = rmse(jack, 14:16)*rmse(jack, 14:16)';
    
    error_Tg = rmse(jack, 17:25)*rmse(jack, 17:25)';
    error_Ts = rmse(jack, 26:34)*rmse(jack, 26:34)';
    error_Ta = rmse(jack, 35:43)*rmse(jack, 35:43)';
    
    error_p_CB = rmse(jack, 44:46)*rmse(jack, 44:46)';
    error_fxy = rmse(jack, 47:48)*rmse(jack, 47:48)';
    error_cxy = rmse(jack, 49:50)*rmse(jack, 49:50)';
    
    error_k1 = rmse(jack, 51)*rmse(jack, 51)';
    error_k2 = rmse(jack, 52)*rmse(jack, 52)';
    error_p1p2 = rmse(jack, 53:54)*rmse(jack, 53:54)';
    
    error_td = rmse(jack, 55)*rmse(jack, 55)';
    error_tr = rmse(jack, 56)*rmse(jack, 56)';
    index =1;
    errors(i,index) = epoch;
    index= index+1;
    pid180 = 180/pi;
    errors(i,index) = (error_bg/3)^.5*pid180;
    index= index+1;
    errors(i,index) = (error_ba/3)^.5;
    index= index+1;
    errors(i,index) = (error_Tg/9)^.5;
    index= index+1;
    errors(i,index) = (error_Ts/9)^.5;
    index= index+1;
    errors(i,index) = (error_Ta/9)^.5;
    index= index+1;
    errors(i,index) = (error_p_CB/3)^.5*100;
    index= index+1;
    errors(i,index) = (error_fxy/2)^.5;
    index= index+1;
    errors(i,index) = (error_cxy/2)^.5;
    index= index+1;    
    errors(i,index) = error_k1^.5;
    index= index+1;
    errors(i,index) = error_k2^.5;
    index= index+1;
    errors(i,index) = (error_p1p2/2)^.5;
    index= index+1;
    errors(i,index) = error_td^.5*1000;
    index= index+1;
    errors(i,index) = error_tr^.5*1000;
end
end