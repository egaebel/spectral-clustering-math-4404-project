function test_online_eigen_code()

% egaebel_qr_algorithm(A, tolerance, MAX_ITERATIONS)

    % Tridiagonal matrix---------------------------------------------------
    fprintf('\n\nTridiagonal matrix-----------------------------------:\n')
    A = [3 1 0;
         1 3 1;
         0 1 3];
    tolerance = 10^-10;
    MAX_ITERATIONS = 10000;
    
    [a, b, vects] = red2st(A);
    % With tridiagonal matrix passed
    [eigenvalues, eigenvectors] = qrst(a, b, ...
                                        tolerance, MAX_ITERATIONS, vects)
    
    fprintf('\n\nEig results from MATLAB\n')
    [eigenvectors, eigenmatrix] = eig(A)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Non tridiagonal matrix-----------------------------------------------
    fprintf('\n\nNon-tridiagonal matrix-------------------------------:\n')
    A = [4 1 -2 2;
         1 2 0 1;
         -2 0 3 -2;
         2 1 -2 -1]
     
    fprintf('\n\nQR Algorithm results\n')

    %function [a, b, V] = red2st ( A )
    [a, b, vects] = red2st(A);
    a
    b
    vects
    % With tridiagonal matrix passed
    [eigenvalues, eigenvectors] = qrst(a, b, tolerance, MAX_ITERATIONS, vects)

    fprintf('\n\nEig results of A from MATLAB\n')
    [eigenvectors, eigenmatrix] = eig(A)
    
    fprintf('\n\nSymmetric Tridiagonal Matrix from Textbook:\n')
    [4 -3 0 0;
     -3 (10 / 3) -(5 / 3) 0;
     0 -(5 / 3) -(33 / 25) (68 / 75);
     0 0 (68 / 75) (149 / 75)]
end

function [lambda, v] = qrst (a, b, TOL, Nmax, vects )

%QRST           determine all of the eigenvalues (and optionally all of the
%               eigenvectors) of a symmetric tridiagonal matrix using the 
%               QR algorithm with Wilkinson shift
%
%     calling sequences:
%             [lambda, v] = qrst ( a, b, TOL, Nmax, vects )
%             [lambda, v] = qrst ( a, b, TOL, Nmax )
%             lambda = qrst ( a, b, TOL, Nmax )
%             qrst ( a, b, TOL, Nmax, vects )
%             qrst ( a, b, TOL, Nmax )
%
%     inputs:
%             a       vector containing elements along the main diagonal
%                     of the symmetric tridiagonal matrix whose eigenvalues
%                     are to be determined
%             b       vector containing elements along the off diagonal
%                     of the symmetric tridiagonal matrix whose eigenvalues
%                     are to be determined
%             TOL     convergence tolerance
%             Nmax    maximum number of iterations
%             vects   optional input argument
%                     matrix containing eigenvector information produced
%                     during the reduction of the original symmetric
%                     matrix to symmetric tridiagonal form
%                     - this input is needed only if computation of the 
%                       eigenvectors is requested (by including the second
%                       output argument) and the original matrix was not in
%                       symmetric tridiagonal form
%
%     output:
%             lambda  vector containing the eigenvalues of the symmetric
%                     tridiagonal matrix determined by the vectors a and b
%             v       optional output argument
%                     matrix containing the eigenvectors of the symmetric
%                     tridiagonal matrix determined by the vectors a and b
%                     - the i-th column of this matrix is an eigenvector
%                       which corresponds to the i-th eigenvalue in the
%                       vector lambda
%                     - eigenvectors will be not computed if this second
%                       output argument is omitted
%
%     NOTE:
%             if the maximum number of iterations is exceeded, a message
%             to this effect will be displayed, along with the number of
%             eigenvalues which had been determined - these eigenvalues
%             will be returned in the last entries of the output vector
%             lambda
%

    n = length(a);

    if ( length(b) == n-1 ) b(2:n) = b(1:n-1); end;

    c = zeros ( 1, n );
    s = zeros ( 1, n );
    shift = 0;
    togo = n;

    if ( nargout >= 2 )
       if ( nargin >= 5 )
          v = vects;
       else
          v = eye(n);
       end;
    end;

    for its = 1 : Nmax

        if ( togo == 1 )
           lambda(1) = a(1) + shift;
           %disp ( its );
           return;
        end;

        trace = a(togo-1) + a(togo);
        det   = a(togo-1)*a(togo) - b(togo)*b(togo);
        disc  = sqrt ( trace*trace - 4*det );
        mu1 = (1/2) * ( trace + disc );
        mu2 = (1/2) * ( trace - disc );
        if ( abs ( mu1 - a(togo) ) < abs ( mu2 - a(togo) ) )
           s = mu1;
        else
           s = mu2;
        end;

        shift = shift + s;
        for i = 1:togo 
            a(i) = a(i) - s;
        end;

        oldb = b(2);
        for i = 2:togo
            j = i-1;
            r = sqrt ( a(j)^2 + oldb^2 );
            c(i) = a(j) / r;
            s(i) = oldb / r;
            a(j) = r;
            temp1 = c(i)*b(i) + s(i)*a(i);
            temp2 = -s(i)*b(i) + c(i)*a(i);
            b(i) = temp1;
            a(i) = temp2;
            if ( i ~= togo ) oldb = b(i+1); b(i+1) = c(i)*b(i+1); end;
        end;

        a(1) = c(2)*a(1) + s(2)*b(2);
        b(2) = s(2)*a(2);
        for i = 2:(togo-1)
            a(i) = s(i+1)*b(i+1) + c(i)*c(i+1)*a(i);
            b(i+1) = s(i+1)*a(i+1);
        end;
        a(togo) = c(togo)*a(togo);

        if ( nargout >= 2 )
           for i = 2 : togo
               col1 = v(:,i-1) * c(i) + v(:,i) * s(i);
               v(:,i) = -s(i) * v(:,i-1) + c(i) * v(:,i);
               v(:,i-1) = col1;
           end;
        end;

        if ( abs(b(togo)) < TOL )
           lambda(togo) = a(togo) + shift;
           %disp([lambda(togo) its]);
           togo = togo - 1;
        end;

    end;

    %disp ( 'qrst error: Maximum number of iterations exceeded' );
    %disp ( sprintf ( '%d eigenvalues determined \n', n-togo ) );
end

function [a, b, V] = red2st ( A )
%RED2ST         perform similarity transformations to reduce the symmetric
%               matrix A to symmetric tridiagonal form
%
%     calling sequences:
%             [a, b, V] = red2st ( A )
%             [a, b] = red2st ( A )
%             red2st ( A )
%
%     input:
%             A       square symmetric matrix to be reduced to symmetric
%                     tridiagonal form
%
%     outputs:
%             a       vector containing elements along the main diagonal
%                     of the symmetric tridiagonal from of A
%             b       vector containing elements along the off diagonal
%                     of the symmetric tridiagonal from of A
%             V       optional output argument
%                     matrix containing eigenvector information for the
%                     matrix A
%

    [nrow, ncol] = size ( A );
    if ( nrow ~= ncol )
       %disp ( 'red2st error: square matrix required' );
       return;
    end;
    n = nrow;

    if ( nargout >= 3 ) V = eye(n); end;

    for i = 1 : n-2
        w = zeros ( n, 1 );
        x = A(:,n-i+1);
        alpha = - sign(x(n-i)) * norm ( x(1:n-i) );
        if ( alpha ~= 0 )
           w(n-i) = sqrt ( (1/2) * ( 1 - x(n-i)/alpha ) );
           w(1:n-i-1) = -(1/2) * x(1:n-i-1) / ( alpha * w(n-i) );

           u = A * w;
           K = dot ( w, u );
           q = u - K * w;
           A = A - 2*w*q' - 2*q*w';

           if ( nargout >= 3 ) 
               V = V - 2*V*w*w';  
           end;	
        end;
    end;

    a = diag(A);
    b = zeros ( n, 1 );
    for i = 2:n b(i) = A(i,i-1); end;
    V
end

