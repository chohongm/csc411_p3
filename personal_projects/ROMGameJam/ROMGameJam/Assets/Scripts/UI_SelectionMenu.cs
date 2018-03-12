using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using UnityEditor.Animations;

// This class handles the player selecting which organism to play as
// TO ADD: proper canvas hiding, better buttons, etc
public class UI_SelectionMenu : MonoBehaviour {

    private GameController gameController;
    public Canvas canvas;
    public GameObject player;
    public AnimatorController[] controllers; // 0 = anomalocaris, 1 = nectocaris, 2 = opabinia, 3 = metaspriggina
    public Sprite[] idleSprites;
    public Button[] buttons;

    void Start()
    {
        player.SetActive(false);
        gameController = GameObject.FindWithTag("GameController").GetComponent<GameController>();
    }

    /*
     *  Attaches the appropriate animator controller, based on the array indices,
     *  to the player GameObject.
     */
    public void SelectPlayer(int index)
    {
        //Debug.Log("selected!");
        player.SetActive(true);
        player.GetComponent<SpriteRenderer>().sprite = idleSprites[index];
        player.GetComponent<Animator>().runtimeAnimatorController = controllers[index];
        gameController.SetCreatureSelected(true);
        canvas.enabled = false;        

        for(int i = 0; i < buttons.Length; i++) {
            buttons[i].interactable = false;
        }
    }
}
