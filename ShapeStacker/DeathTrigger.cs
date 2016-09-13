using UnityEngine;
using System.Collections;

public class DeathTrigger : MonoBehaviour {

    void OnTriggerEnter2D(Collider2D col)
    {
        //hasLost = true;
        int result;
        if (int.TryParse(col.gameObject.GetComponent<TextMesh>().text, out result))
        {
            if (result % BlackHole.rand == 0)
            {
                GameObject.FindObjectOfType<ScoreManager>().score--;
                Debug.Log(result);
            }
        }
        Destroy(col.gameObject);
    }
}
