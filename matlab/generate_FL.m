function generate_FL(prefix,howmany)
i=1;
for xx=0:0.2:1
    for yy=0:0.2:1
        for zz=0:0.2:1
            X(i)=xx;
            Y(i)=yy;
            Z(i)=zz;
            i=i+1;
        end
    end
end
max_x=640;
max_y=480;

P_X=[X' Y' Z' ones(size(X))']';
cur=1;
while (cur<=howmany)

    %distortion model. Hardcoded for now
    A=rand/2;
    B=rand/2;
    C=rand/2;
    %[f ox oy rx ry rz tx ty tz]
    x0=[1000+(rand-0.5)*250 rand*640 rand*480 rand rand rand rand*10 rand*10 rand*10];
    %finds a suitable set of camera parameters
    [P oP]=fminsearch(@f_fits,x0);
    if ((oP==0)&&(P(9)>0))
    
        %generates the .ref for scene cur
        f=fopen([prefix '_' sprintf('%02d',cur) '.ref'],'w');
        for i=1:size(P_X,2)
            fprintf(f,'%d %.5f %.5f %.5f\n',i-1,P_X(1,i),P_X(2,i),P_X(3,i));
        end
        fclose(f);
    

        %save them for future reference
        f=fopen([prefix '_' sprintf('%02d',cur) '.tca'],'w');
        fprintf(f,'%.5g ',[P A B C]);
        fclose(f);
    
        %save the projected calibration points for scene cur
        x=gp_proj(P);
        %figure,plot(x(1,:),x(2,:),'ored');

        f=fopen([prefix '_' sprintf('%02d',cur) '.dat'],'w');
        xp=reshape(x(1:2,:),[1 2*size(x,2)]);
        fprintf(f,'0 ');
        fprintf(f,'%.5g ',xp);
        fclose(f);
    
        cur=cur+1;
    else
        sprintf('got a wrong one! %d',oP);
    end
    
end

    function x=gp_proj(P)
        %[f ox oy rx ry rz tx ty tz]
        K=[[P(1) 0.01 P(2)]; [0 P(1) P(3)]; [0 0 1]];
        R=rodrigues([P(4) P(5) P(6)]);
        T=[P(7) P(8) P(9)]';
       % A=P(10);B=P(11);C=P(12);

        x=[R T]*P_X;
        x=x./repmat(x(3,:),[3 1]);

        x(1:2,:)=x(1:2,:)-repmat(mean(x(1:2,:)')',[1 size(x,2)]);  %#ok<*UDIM>

        %xt=K*x1;
        %figure(1),plot(xt(1,:),xt(2,:),'xblue'); hold on

        a=x(1,:);
        b=x(2,:);
        r2=a.^2+b.^2;

        a= a - a.*A.*r2 - B.*(r2+2*a) - C.*a.*b;
        b= b - b.*A.*r2 + B.*(r2+2*b) + C.*a.*b;
%        a=a-A*a;
%        b=b+B*b;

        x(1,:)=a;
        x(2,:)=b;
        x=K*x;
                
        %figure(1),plot(x1(1,:),x1(2,:),'ored');axis tight

    end

    function offPoints=f_fits(P)
        x=gp_proj(P);
        offPoints=sum(x(1,:)<0)+sum(x(1,:)>max_x)+sum(x(2,:)<0)+sum(x(2,:)>max_y);
    end
end
