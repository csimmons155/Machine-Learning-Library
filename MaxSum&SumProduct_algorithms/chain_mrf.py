import math

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        #print "VALES...",i,a,b,self._n
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            print "ERROR: VALES...",i,a,b
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.sum = 0
        # TODO: EDIT HERE
        # add whatever data structures needed

    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0

        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0

        result = [1.0 / (self._potentials.num_x_values())] * (self._potentials.num_x_values() + 1)
        result[0] = 0
        prob = 0

        #loop through all possible k values
        for i in range(1, int(self._potentials.num_x_values()) + 1):
            curr_n = int(self._potentials.chain_length())
            #print "Current Length", curr_n
            #quit()
            #print "x_i", x_i, "has k value:", i
            #for the first node
            if x_i == 1:
                # curr_n +1 = factor node
                # x_i is our current node, i = the k value of that node
                prob = self.unary_factor(x_i, i) * self.binary_message1(curr_n+1, x_i, i, x_i + 1)

            if x_i == int(self._potentials.chain_length()):
                n = (curr_n - 1) + x_i
                prob = self.unary_factor(x_i, i)* self.binary_message1(n, x_i, i, x_i - 1)

            if x_i != 1 and x_i != int(self._potentials.chain_length()):
                f_left = (curr_n - 1) + x_i
                f_right = curr_n + x_i
                sec_fac = self.binary_message1(f_left, x_i , i, x_i - 1)
                third_fac = self.binary_message1(f_right, x_i, i, x_i + 1)
                prob = self.unary_factor(x_i,i) * sec_fac * third_fac

            #result.append(prob)
            result[i] = prob

        sum1 = sum(result)
        self.sum = sum1
        for k in range(len(result)):
            result[k] = result[k] / sum1

        return result

    def getSum(self):
        return self.sum

    def unary_factor(self, x_i, a):
        #print "unary: x_i", x_i
        return self._potentials.potential(x_i,a)

    def binary_message1(self, f_s, curr_xi, a, next_xi ):
        ret_message = 0

        #for loop from 1 to k
        for j in range(1, int(self._potentials.num_x_values()) + 1):
            #print "binary_mess1: factor f_s=",f_s, "with x_i", curr_xi, " of value:", a, "next x_i =", next_xi, "of value:", j

            #next_xi the neighboring xi; j = the value of that xi, the current factor: f_s
            if (next_xi < curr_xi):
                #print "Why..."

                #changed a and j -> self.binary_factor(f_s, a, j)
                ret_message += self.binary_factor(f_s, j, a) * self.binary_message2(next_xi, j, f_s, backward= True)
            else:
                ret_message += self.binary_factor(f_s, a, j) * self.binary_message2(next_xi, j, f_s)

        return ret_message

    def binary_message2(self, x_i, a, f_s, backward = False):
        if backward: new_fs = f_s - 1
        else: new_fs = f_s + 1

        #print "binary_message2: x_i (next_xi):",x_i,"with value:", a, "and factor:", f_s, "(backward):", backward

        if x_i == int(self._potentials.chain_length()):
            #if x_i is last node: x_n

            ret_message = self.unary_factor(x_i, a)

        elif x_i == 1:
            ret_message = self.unary_factor(x_i, a)


        else:
            #if x_i is not the last node

            if backward:
                ret_message = self.unary_factor(x_i, a) * self.binary_message1(new_fs, x_i, a, x_i - 1)
            else:
                ret_message = self.unary_factor(x_i, a) * self.binary_message1(new_fs, x_i, a, x_i + 1)

        return ret_message


    def binary_factor(self,factor_num, a, b):
        #print "binary_factor lookup: f_s:", factor_num, "and values i,j: ", a, b
        return self._potentials.potential(factor_num, a, b)


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        max_prob = -1000
        prob = 0
        best_xi_value = -1000

        sp = SumProduct(self._potentials)
        v = sp.marginal_probability(x_i)
        our_sum = sp.getSum()



        for i in range(1,self._potentials.num_x_values() +1):
            curr_n = self._potentials.chain_length()
            if x_i == 1:
                prob = self.unary_factor(x_i,i) + self.binary_message1(curr_n+1, x_i, i, x_i + 1)


            if x_i == self._potentials.chain_length():
                n = (curr_n - 1) + x_i
                prob = self.unary_factor(x_i, i) + self.binary_message1(n, x_i, i, x_i - 1)


            if x_i != 1 and x_i != self._potentials.chain_length():
                f_left = (curr_n - 1) + x_i
                f_right = curr_n + x_i
                sec_fac = self.binary_message1(f_left, x_i , i, x_i - 1)
                third_fac = self.binary_message1(f_right, x_i, i, x_i + 1)
                prob = self.unary_factor(x_i,i) + sec_fac + third_fac


            if prob > max_prob:
                best_xi_value = i
                max_prob = prob

        #max_prob = math.log(max_prob,2)
        #print "THE SUM", math.log(our_sum,2)
        self._assignments[x_i] = best_xi_value
        return max_prob - math.log(our_sum)
        #return 0.0

    def binary_message1(self, f_s, curr_xi, a, next_xi):
        ret_message = -1000
        best_next_xi_val = -1000
        for j in range(1,self._potentials.num_x_values() + 1):
            if (next_xi < curr_xi):
                #changed a,j here too
                test_message = self.binary_factor(f_s, j, a) + self.binary_message2(next_xi, j, f_s, backward= True)
                if test_message > ret_message:
                    ret_message = test_message
                    best_next_xi_val = j
            else:
                test_message = self.binary_factor(f_s, a, j) + self.binary_message2(next_xi, j, f_s)
                if test_message > ret_message:
                    ret_message = test_message
                    best_next_xi_val = j

        self._assignments[next_xi] = best_next_xi_val
        return ret_message


    def binary_message2(self,x_i, a, f_s, backward = None):
        if backward: new_fs = f_s - 1
        else: new_fs = f_s + 1

        if x_i == self._potentials.chain_length():
            #if x_i is last node: x_n
            ret_message = self.unary_factor(x_i, a)

        elif x_i == 1:
            ret_message = self.unary_factor(x_i, a)

        else:
            #if x_i is not the last node
            if backward:
                ret_message = self.unary_factor(x_i, a) + self.binary_message1(new_fs, x_i, a, x_i - 1)
            else:
                ret_message = self.unary_factor(x_i, a) + self.binary_message1(new_fs, x_i, a, x_i + 1)

        return ret_message


    def binary_factor(self, factor_num, a, b):
        return math.log(self._potentials.potential(factor_num, a, b))


    def unary_factor(self, x_i, a):
        return math.log(self._potentials.potential(x_i,a))




