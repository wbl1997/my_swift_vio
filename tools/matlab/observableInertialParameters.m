function observableInertialParameters()

% parameters: bg ba, Tg, Ts, Ta,
% measurements: 
% wm= Tg*w+Ts*a+bg+n1(1)
% am= Ta*a +ba+ n2(2)
% w_tilde = w+ n3;(3)
% a_tilde = a +n4;(4)
% substitute (3) and (4) into (1)(2)
num=10;
H= zeros(6*num, 33);
for i=1:num
    omega= rand(3,1);
    acc= rand(3,1);
subH= [eye(3), zeros(3), dT_dvecT_v(omega), dT_dvecT_v(acc), zeros(3,9);...
    zeros(3), eye(3), zeros(3,18), dT_dvecT_v(acc)];
H(i*6-5:i*6, :)= subH;
end
rankH= rank(H)
expectedRank=33
    function product = dT_dvecT_v(v)
        assert(size(v,1)==3 &&size(v,2)==1);
        product =[v', zeros(1,6); zeros(1,3), v', zeros(1,3); zeros(1,6), v'];
    end
end