using UnityEngine;
using System.Collections;

public class ScoreManager : MonoBehaviour {

    public int score = 0;
    public bool hasLost = false;
    public Transform explosionPrefab;
	
	// Update is called once per frame
	void Update () {
	    if (score <= -10)
        {
            //hasLost = true;
        }
	}

    void OnGUI()
    {
        GUI.color = Color.black;
        GUI.Label(new Rect(0, 0, 100, 50), "Score: " + score);

        if (hasLost)
        {
            GUI.color = Color.black;
            GUI.Label(new Rect(Screen.width / 2 - 50, Screen.height / 2 - 25, 100, 50), "GAME OVER!");
            if (GUI.Button(new Rect(Screen.width / 2 - 50, Screen.height / 2 - 25 + 55, 100, 50), "Restart!"))
            {
                Application.LoadLevel(Application.loadedLevel);
            }
        }
    }
}
