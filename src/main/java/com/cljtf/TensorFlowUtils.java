package com.cljtf;

import java.lang.reflect.Array;

import org.tensorflow.DataType;

/**
 * Utilities
 * 
 * @author feldi
 *
 * Adapted from /tensorflow/java/src/test/java/org/tensorflow/TestUtil.java
 */
public class TensorFlowUtils {

	// prevent instantiation
	private TensorFlowUtils() {
	}

	/**
	 * Counts the total number of elements in an ND array.
	 * 
	 * @param array
	 *            the array to count the elements of
	 * @return the number of elements
	 */
	public static int flattenedNumElements(Object array) {
		int count = 0;
		for (int i = 0; i < Array.getLength(array); i++) {
			Object e = Array.get(array, i);
			if (!e.getClass().isArray()) {
				count += 1;
			} else {
				count += flattenedNumElements(e);
			}
		}
		return count;
	}

	/**
	 * Flattens an ND-array into a 1D-array with the same elements.
	 * 
	 * @param array
	 *            the array to flatten
	 * @param elementType
	 *            the element class (e.g. {@code Integer.TYPE} for an
	 *            {@code int[]})
	 * @return a flattened array
	 */
	public static Object flatten(Object array, Class<?> elementType) {
		Object out = Array.newInstance(elementType, flattenedNumElements(array));
		flatten(array, out, 0);
		return out;
	}

	private static int flatten(Object array, Object out, int next) {
		for (int i = 0; i < Array.getLength(array); i++) {
			Object e = Array.get(array, i);
			if (!e.getClass().isArray()) {
				Array.set(out, next++, e);
			} else {
				next = flatten(e, out, next);
			}
		}
		return next;
	}

	/**
	 * Converts a {@code boolean[]} to a {@code byte[]}.
	 *
	 * <p>
	 * Suitable for creating tensors of type {@link DataType#BOOL} using
	 * {@link java.nio.ByteBuffer}.
	 */
	public static byte[] bool2byte(boolean[] array) {
		byte[] out = new byte[array.length];
		for (int i = 0; i < array.length; i++) {
			out[i] = array[i] ? (byte) 1 : (byte) 0;
		}
		return out;
	}

	/**
	 * Converts a {@code byte[]} to a {@code boolean[]}.
	 *
	 * <p>
	 * Suitable for reading tensors of type {@link DataType#BOOL} using
	 * {@link java.nio.ByteBuffer}.
	 */
	public static boolean[] byte2bool(byte[] array) {
		boolean[] out = new boolean[array.length];
		for (int i = 0; i < array.length; i++) {
			out[i] = array[i] != 0;
		}
		return out;
	}

}
