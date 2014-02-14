function [gp]=gpcal_config(gp)
%REGRESSMULTI GPTIPS configuration file demonstrating multiple gene symbolic 
%regression on data (y) generated from a non-linear function of 4 inputs (x1, x2, x3, x4).
%
%   [GP]=GPDEMO2_CONFIG(GP) returns a parameter structure GP containing the settings
%   for GPDEMO2 (multigene symbolic regression using genetic programming). 
% 
%   Remarks:
%   There is one output y which is a non-linear
%   function of the four inputs y=exp(2*x1*sin(pi*x4)) + sin(x2*x3).
%   The objective of a GP run is to evolve a multiple gene symbolic
%   function of x1, x2, x3 and x4 that closely approximates y.
%
%   This function was described by:
%   Cherkassky, V., Gehring, D., Mulier F, Comparison of adaptive methods for function
%   estimation from samples, IEEE Transactions on Neural Networks, 7 (4), pp. 969-
%   984, 1996. (Function 10 in Appendix 1)
%   
%   Example:
%   [GP]=RUNGP('gpdemo2_config') uses this configuration file to perform symbolic
%   regression with multiple gene individuals on the data from the above function.   
%
%   (C) Dominic Searson 2009
% 
%   v1.0
%
%   See also: GPDEMO2, GPDEMO3_CONFIG, GPDEMO4, GPDEM03, GPDEMO1, REGRESSMULTI_FITFUN



% Main run control parameters
% --------------------------
gp.runcontrol.pop_size=100;                     % Population size
gp.runcontrol.num_gen=10;				        % Number of generations to run for including generation zero (i.e. if set to 100, it'll finish after generation 99).
k_clusters=20;
howmany=8;


gp.runcontrol.verbose=1;                       % Set to n to display run information to screen every n generations 




% Selection method options
% -------------------------
gp.selection.method='tour';                 % Only tournament selection is currently supported.											 
gp.selection.tournament.size=3;                                                                                         
gp.selection.tournament.lex_pressure=true;  % True to use Luke and Panait's plain lexicographic tournament selection

 
             

% Fitness function specification 
% -------------------------------
gp.fitness.fitfun=@gpcal_fitness;    
% Function handle of the fitness function (filename with no .m extension).

gp.fitness.minimisation=true;        
% Set to true if you want to minimise the fitness function (if false it is maximised).
gp.fitness.terminate=true;
gp.fitness.terminate_value=0.1;



% Set up user data
% ----------------
%%
%load in the raw x and y data
wCases=[1920 1920 1280 1280 1920 1280 3840 1280];
hCases=[1080 1080  720  720 1440  720 2160 960];



for cur=1:howmany
    dat=load(sprintf('%g.dat',cur));
    x=zeros(2*size(dat,1),size(dat,2)/2);
    for i=1:size(dat,1)
        x(2*(i-1)+1,:)=dat(i,1:2:end);
        x(2*(i-1)+2,:)=dat(i,2:2:end);
    end

    ref=load(sprintf('%g.ref',cur));
    X=zeros(2*size(ref,1),size(ref,2)/3);
    for i=1:size(ref,1)
        X(2*(i-1)+1,:)=ref(i,1:3:end);
        X(2*(i-1)+2,:)=ref(i,2:3:end);
    end

    [gp.userdata.A(cur).A gp.userdata.OT(cur).OT gp.userdata.F(cur).usedFrames]=PrepareCalibration(X,x,hCases(cur),wCases(cur),k_clusters);
    gp.userdata.X(cur).X=X;
    gp.userdata.x(cur).x=x;
end

%gp.userdata.initpop=
% radTan={'plus(x8,x9)'
%     'times(times(x8,x9),r2uv(x8,x9))'
%     'times(times(x8,x9),p2(r2uv(x8,x9)))'
%     'plus(times(3,p2(x8)),swap(p2(x9)))'
%     'plus(times(3,p2(x9)),swap(p2(x8)))'
%     'plus(times(x8,swap(x9)),times(x9,swap(x8)))'
%     'r2uv(x8,x9)'}';
% 
% polyFish={'plus(x8,x9)'
%     'ruv(x8,x9)'
%     'r2uv(x8,x9)'
%     'p3(ruv(x8,x9))'
%     'p2(r2uv(x8,x9))'}';

radTan={'c(x8,x9)'
    'a(a(x8,x9),k(x8,x9))'
    'a(a(x8,x9),m(k(x8,x9)))'
    'c(a(3,m(x8)),l(m(x9)))'
    'c(a(3,m(x9)),l(m(x8)))'
    'c(a(x8,l(x9)),a(x9,l(x8)))'
    'k(x8,x9)'}';

polyFish={'c(x8,x9)'
    'h(x8,x9)'
    'k(x8,x9)'
    'n(h(x8,x9))'
    'm(k(x8,x9))'}';


gp.userdata.initpop={{'c(x8,x9)'},radTan,polyFish};

            




% Input configuration
% -------------------   
% This sets the number of inputs (i.e. the size of the terminal set NOT including constants)
gp.nodes.inputs.num_inp=9;% max 9
%--->crap way of adapting for symbolic constants. inputs 8/9 are real, the rest are symbolic constants


gp.nodes.const.p_ERC=0;            
% we don't need constants, not that kind anyway

% Tree build options
% ----------------------

                         
gp.treedef.max_depth=5;                    
% Maximum depth of trees                     

                                        
                                    

% Multiple gene settings
% ----------------------
gp.genes.multigene=true;
% True=use multigene individuals False=use ordinary single gene individuals.
gp.genes.max_genes=8;% max: 8
% The absolute maximum number of genes allowed in an individual.                              

    


% Define functions
% ----------------  
%   (Below are some definitions of functions that have been used for symbolic regression problems) 

%         Function name (must be an mfile or builtin function on the path).
      
gp.nodes.functions.name{1}='times'      ;           
gp.nodes.functions.name{2}='minus'      ;               
gp.nodes.functions.name{3}='plus'       ;             
gp.nodes.functions.name{4}='pdivide'    ;            
gp.nodes.functions.name{5}='u'          ;
gp.nodes.functions.name{6}='v'          ;
gp.nodes.functions.name{7}='ruv'        ;            
gp.nodes.functions.name{8}='r2uv'       ;
gp.nodes.functions.name{9}='swap'       ;
gp.nodes.functions.name{10}='p2'        ;            
gp.nodes.functions.name{11}='p3'        ;            
%gp.nodes.functions.name{7}='pumean'    ;
%gp.nodes.functions.name{8}='pumean'    ;



% Active functions
% ----------------
%
% Manually setting a function node to inactive allows you to exclude a function node in a 
% particular run.
%  gp.nodes.functions.active(1)=1;                          
%  gp.nodes.functions.active(2)=1;                          
%  gp.nodes.functions.active(3)=1;                          
%  gp.nodes.functions.active(4)=1;                          
%  gp.nodes.functions.active(5)=1;                          
%  gp.nodes.functions.active(6)=1;                          
%  gp.nodes.functions.active(7)=1;                           
%  gp.nodes.functions.active(8)=0;                           
%  gp.nodes.functions.active(9)=0;                         
%  gp.nodes.functions.active(10)=0;                         
%  gp.nodes.functions.active(11)=0;                         
% gp.nodes.functions.active(12)=0;  
% gp.nodes.functions.active(13)=0;

                                     
                                   

