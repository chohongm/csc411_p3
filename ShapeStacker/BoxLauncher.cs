using UnityEngine;
using System.Collections;

public class BoxLauncher : MonoBehaviour {

    public GameObject objectPrefab;
    //public string currLauncher = "LeftLauncher";

    public float fireDelay = 6.76f; //6.76 for triforce music, 7.164 for techno music
    public float fireVelocity = 22f;
    public float cooldown = 1.6f;
    public float nextFire = 0f;

    public float spawnThreshold = 0.4f;
    public int frequency = 3;
    public FFTWindow fftWindow;
    public float debugValue;

    private float[] samples = new float[1024]; //MUST BE A POWER OF TWO

    /*
    // for moving the launcher as camera goes up
    void Update()
    {
        Vector3 pos = transform.position;
        float camYChange = Camera.main.GetComponent<CameraMover>().targetY - 
                           Camera.main.GetComponent<CameraMover>().transform.position.y;
        pos.y = Mathf.Lerp(transform.position.y, transform.position.y + camYChange, 1 * Time.deltaTime);
        transform.position = pos;
    }
    */

    void FixedUpdate()
    {
        if (nextFire > 0)
        {
            nextFire -= Time.deltaTime;
        } else
        {
            nextFire = fireDelay;
        }
        cooldown -= Time.deltaTime;
    }

    // Update is called once per frame
    void Update() {

        if (GameObject.FindObjectOfType<ScoreManager>().hasLost)
        {
            return;
        }

        AudioListener.GetSpectrumData(samples, 0, fftWindow);

        debugValue = samples[frequency];

        if (nextFire <= 3.38) // 3.38 for triforce music, 3.582 for techno
        {
            if (cooldown <= 0 && samples[frequency] > spawnThreshold)
            {
                cooldown = 1.6f;
                GameObject copy;
                if (gameObject.transform.name == "TopLauncher")
                {
                    copy = (GameObject)Instantiate(
                    objectPrefab,
                    transform.position,
                    transform.rotation * Quaternion.Euler(0, 0, 180));
                } else
                {
                    copy = (GameObject)Instantiate(
                    objectPrefab,
                    transform.position,
                    transform.rotation);
                }
                copy.GetComponent<Rigidbody2D>().velocity = transform.rotation * new Vector2(0, fireVelocity);
            }
        }
	}
}
