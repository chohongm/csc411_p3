using UnityEngine;
using System.Collections;

public class BlackHole : MonoBehaviour {

    public float divisorDelay = 13.52f;
    public float nextDivisor = 0f;
    public CircleCollider2D collider;

    public static int rand;

    float threshold;
    float scoreRadius;

    void OnTriggerEnter2D(Collider2D col)
    {
        int entered;
        
        if (GameObject.FindObjectOfType<ScoreManager>().hasLost)
        {
            return;
        }

        if (int.TryParse(col.gameObject.GetComponent<TextMesh>().text, out entered))
        {
            if (entered % rand == 0)
            {
                GameObject.FindObjectOfType<ScoreManager>().score++;
            }
            else
            {
                GameObject.FindObjectOfType<ScoreManager>().score--;
            }
        }
        Destroy(col.gameObject);
    }
	
	// Update is called once per frame
	void FixedUpdate () {

        if (GameObject.FindObjectOfType<ScoreManager>().hasLost)
        {
            return;
        }

        threshold = FindObjectOfType<BoxLauncher>().debugValue;
        collider = transform.GetComponent<CircleCollider2D>();
        int score = FindObjectOfType<ScoreManager>().score;
        if (score > 0)
        {
            scoreRadius = score / 100;

        } else
        {
            scoreRadius = 0;
        }

        collider.radius = 0.6f + (threshold / 2f) + scoreRadius;

        nextDivisor -= Time.deltaTime;

        if (nextDivisor <= 0)
        {
            nextDivisor = divisorDelay;
            rand = ((int)Random.Range(1, 9));
            gameObject.GetComponent<TextMesh>().text = rand.ToString();
        }
    }
}
