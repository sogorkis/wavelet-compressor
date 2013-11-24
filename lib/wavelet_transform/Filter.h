/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef FILTER_H
#define FILTER_H

/**
  * Class encapsulating signal processing filter data. Filter is described by three
  * basic attributes:
  * <ul>
  * <li>filter values (float *)</li>
  * <li>filter length (int)</li>
  * <li>filter first index (int) - must be between [0, filer length)</li>
  * </ul>
  */
class Filter {
public:
    /**
      * Constructs filter with specified values.
      * @param values filter values
      * @param length filter length
      * @param firstIndex fileter first index
      */
    Filter(const float * values, int length, int firstIndex = 0);

    ~Filter();

    /**
      * Returns filter values.
      * @return filter values.
      */
    const float * getValues() const { return values; }

    /**
      * Returns filter length.
      * @return filter length.
      */
    int getLength() const { return length; }

    /**
      * Returns filter first index.
      * @return filter first index.
      */
    int getFirstIndex() const { return firstIndex; }
private:
    float * values;
    int length;
    int firstIndex;
};

#endif // FILTER_H
