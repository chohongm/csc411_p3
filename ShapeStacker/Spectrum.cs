using UnityEngine;
using System.Collections;

public class Spectrum : MonoBehaviour {

    public float[] freqData = new float[8192];
 
    public float[] band;
    public GameObject[] g;
 
    void Start()
    {
        int n = freqData.Length;
        int k = 0;
        for (var j = 0; j < freqData.Length; j++)
        {
            n /= 2;
            if (n == 0) break;
            k++;
        }
        band = new float[k + 1];
        g = new GameObject[k + 1];
        for (var i = 0; i < band.Length; i++)
        {
            band[i] = 0;
            g[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            g[i].transform.position = new Vector3(i-10, 0, 0);
        }
        InvokeRepeating("check", 0, 1f / 15f); // update at 15 fps
    }

    void check()
    {
        AudioListener.GetSpectrumData(freqData, 0, FFTWindow.Rectangular);

        int k = 0;
        int crossover = 2;
        for (int i = 0; i < freqData.Length; i++)
    {
            var d = freqData[i];
            var b = band[k];
            band[k] = (d > b) ? d : b;   // find the max as the peak value in that frequency band.
            if (i > crossover - 3)
            {
                k++;
                crossover *= 2;   // frequency crossover point for each band.
                Vector2 newPos = g[k].transform.position;
                newPos.y = band[k] * 32;
                g[k].transform.position = newPos;
                band[k] = 0;
            }
        }
    }
    //private float sqrt = Mathf.Sqrt();
}
