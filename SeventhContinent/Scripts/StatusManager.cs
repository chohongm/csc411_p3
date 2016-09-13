using UnityEngine;
using System.Collections;

public class StatusManager : MonoBehaviour {

    public string civName;
    public int gold;
    public int population;
    public int fame;
    public int food;
    public int lumber;
    public int stone;
    public int numTotWorkers;
    public int numIdleWorkers;
    //public Text status;
    private UnityEngine.UI.Text status;

    private float foodConsumeInterval;
    private float timePassed;


    // Use this for initialization
    void Start() {
        transform.SetAsLastSibling();
        //status = gameObject.GetComponentInChildren<Text>();
        status = gameObject.GetComponent<UnityEngine.UI.Text>();
        foodConsumeInterval = 30;
        timePassed = 0;
    }

    // Update is called once per frame
    void Update() {
        if (timePassed >= foodConsumeInterval) {
            food -= (int) Mathf.Floor(population / 5);
            timePassed = 0;
        } else {
            timePassed += Time.deltaTime;
        }
        status.text = civName + "\nPopulation : " + population + "\nFame : " + fame +
            //"\nGold : " + gold + 
            "\nFood : " + food + "\nLumber : " + lumber +
            "\nStone : " + stone + " \nWorker Count : " + numIdleWorkers + " / " + numTotWorkers;
    }

    public void EffectPlus(string type, int value) {
        switch (type) {
            case "Population":
                population += value;
                numTotWorkers++;
                numIdleWorkers++;
                break;
            case "Fame":
                fame += value;
                break;
            case "Food":
                food += value;
                break;
            case "Lumber":
                lumber += value;
                break;
            case "Stone":
                stone += value;
                break;
            default:
                break;
        }
    }

    public void EffectMinus(string type, int value) {
        switch (type) {
            case "Population":
                population -= value;
                // total worker-- and idle worker-- but if no idle worker then randomly remove any assigned worker
                break;
            case "fame":
                fame -= value;
                break;
            case "food":
                food -= value;
                break;
            case "Lumber":
                lumber -= value;
                break;
            case "Stone":
                stone -= value;
                break;
            default:
                break;
        }
    }

    public int GetCurrency(string type) {
        switch (type) {
            case "Lumber":
                return lumber;
            case "Stone":
                return stone;
            default:
                return 0;
        }
    }

    public void activateSwitch() {
        this.GetComponent<UnityEngine.UI.Text>().enabled = !this.GetComponent<UnityEngine.UI.Text>().enabled;
        transform.parent.GetChild(0).GetComponent<UnityEngine.UI.Image>().enabled =
            !transform.parent.GetChild(0).GetComponent<UnityEngine.UI.Image>().enabled;
    }
}
